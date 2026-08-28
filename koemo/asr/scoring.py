"""Calibrated N-best scoring and rank-only LLM integration."""
from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import median, pstdev
from typing import Iterable, Mapping, Sequence

from .schema import RankedHypothesis, TranscriptHypothesis, TranscriptState


@dataclass(frozen=True, slots=True)
class ScoreWeights:
    """Log-linear acoustic/lattice fusion weights.

    Every component is robustly standardized across the current N-best list, so
    these weights are interpretable relative importances instead of attempting
    to mix incompatible raw score scales.
    """

    whisper: float = 1.00
    char_ctc: float = 0.45
    mora_ctc: float = 0.70
    alignment: float = 0.25
    prosody: float = 0.15
    coverage: float = 0.20
    no_speech_penalty: float = 0.25
    compression_penalty: float = 0.10
    compression_threshold: float = 2.40


@dataclass(frozen=True, slots=True)
class LLMRankVote:
    """Schema accepted from a rank-only LLM call.

    The LLM returns candidate IDs, never replacement text.
    """

    candidate_id: str
    rank: int
    confidence: float

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id is required")
        if self.rank < 1:
            raise ValueError("rank is 1-based")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


def _robust_z(values: Sequence[float | None]) -> tuple[float, ...]:
    """Robustly standardize present values; missing values are neutral (0)."""

    present = [value for value in values if value is not None]
    if len(present) <= 1:
        return tuple(0.0 for _ in values)

    center = median(present)
    absolute_deviations = [abs(value - center) for value in present]
    scale = 1.4826 * median(absolute_deviations)
    if scale < 1e-8:
        scale = pstdev(present)
    if scale < 1e-8:
        return tuple(0.0 for _ in values)

    return tuple(0.0 if value is None else (value - center) / scale for value in values)


def _component_vectors(
    hypotheses: Sequence[TranscriptHypothesis],
    weights: ScoreWeights,
) -> Mapping[str, tuple[float, ...]]:
    features = [hypothesis.features for hypothesis in hypotheses]
    compression_quality: list[float | None] = []
    for feature in features:
        if feature.compression_ratio is None:
            compression_quality.append(None)
        else:
            compression_quality.append(
                -max(0.0, feature.compression_ratio - weights.compression_threshold)
            )

    return {
        "whisper": _robust_z([feature.whisper_logprob for feature in features]),
        "charCtc": _robust_z([feature.char_ctc_logprob for feature in features]),
        "moraCtc": _robust_z([feature.mora_ctc_logprob for feature in features]),
        "alignment": _robust_z([feature.alignment_quality for feature in features]),
        "prosody": _robust_z([feature.prosody_quality for feature in features]),
        "coverage": _robust_z([feature.coverage for feature in features]),
        "noSpeech": _robust_z(
            [
                None
                if feature.no_speech_probability is None
                else -feature.no_speech_probability
                for feature in features
            ]
        ),
        "compression": _robust_z(compression_quality),
    }


def rank_acoustic_hypotheses(
    hypotheses: Iterable[TranscriptHypothesis],
    *,
    weights: ScoreWeights = ScoreWeights(),
) -> tuple[RankedHypothesis, ...]:
    """Rank N-best hypotheses using acoustic/lattice evidence only."""

    items = tuple(hypotheses)
    if not items:
        raise ValueError("at least one hypothesis is required")
    candidate_ids = [item.candidate_id for item in items]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate IDs must be unique")

    components = _component_vectors(items, weights)
    component_weights = {
        "whisper": weights.whisper,
        "charCtc": weights.char_ctc,
        "moraCtc": weights.mora_ctc,
        "alignment": weights.alignment,
        "prosody": weights.prosody,
        "coverage": weights.coverage,
        "noSpeech": weights.no_speech_penalty,
        "compression": weights.compression_penalty,
    }

    scored: list[tuple[TranscriptHypothesis, float, dict[str, float]]] = []
    for index, hypothesis in enumerate(items):
        weighted_components = {
            name: components[name][index] * component_weights[name]
            for name in components
        }
        total = sum(weighted_components.values())
        scored.append((hypothesis, total, weighted_components))

    # Deterministic candidate ID tie-break keeps replays reproducible.
    scored.sort(key=lambda item: (-item[1], item[0].candidate_id))
    return tuple(
        RankedHypothesis(
            hypothesis=hypothesis,
            acoustic_score=score,
            rank=rank,
            component_scores=component_scores,
        )
        for rank, (hypothesis, score, component_scores) in enumerate(scored, start=1)
    )


def select_observed_transcript(
    hypotheses: Iterable[TranscriptHypothesis],
    *,
    weights: ScoreWeights = ScoreWeights(),
) -> TranscriptState:
    """Stage 2: freeze ``observedTranscript`` from non-LLM evidence."""

    ranked = rank_acoustic_hypotheses(hypotheses, weights=weights)
    winner = ranked[0].hypothesis
    return TranscriptState(
        observed_transcript=winner.text,
        observed_candidate_id=winner.candidate_id,
        hypotheses=ranked,
        mora_units=winner.mora_units,
    )


def _bounded_borda_score(rank: int, count: int, confidence: float, limit: float) -> float:
    if count <= 1:
        return 0.0
    position = (count - rank) / (count - 1)  # best=1, worst=0
    centered = 2.0 * position - 1.0
    return max(-limit, min(limit, centered * confidence * limit))


def attach_llm_rank_only(
    state: TranscriptState,
    votes: Iterable[LLMRankVote],
    *,
    max_tiebreak_score: float = 0.15,
    minimum_preference_confidence: float = 0.50,
) -> TranscriptState:
    """Stage 3: attach an LLM ranking without rewriting the observation.

    ``observedTranscript`` and ``observedCandidateId`` are intentionally copied
    unchanged.  The bounded tiebreak signal is logged for experiments and can be
    enabled in a future policy only after an explicit ablation proves it safe.
    """

    if max_tiebreak_score < 0.0:
        raise ValueError("max_tiebreak_score must be non-negative")
    if not 0.0 <= minimum_preference_confidence <= 1.0:
        raise ValueError("minimum_preference_confidence must be in [0, 1]")
    vote_items = tuple(votes)
    vote_by_id = {vote.candidate_id: vote for vote in vote_items}
    if len(vote_by_id) != len(vote_items):
        raise ValueError("LLM vote candidate IDs must be unique")

    known_ids = {item.hypothesis.candidate_id for item in state.hypotheses}
    unknown_ids = set(vote_by_id) - known_ids
    if unknown_ids:
        raise ValueError(f"LLM returned unknown candidate IDs: {sorted(unknown_ids)}")

    ranks = [vote.rank for vote in vote_items]
    if len(set(ranks)) != len(ranks):
        raise ValueError("LLM ranks must be unique")
    if any(rank > len(state.hypotheses) for rank in ranks):
        raise ValueError("LLM rank exceeds the number of known candidates")

    count = len(state.hypotheses)
    annotated: list[RankedHypothesis] = []
    for item in state.hypotheses:
        vote = vote_by_id.get(item.hypothesis.candidate_id)
        if vote is None:
            annotated.append(item)
            continue
        annotated.append(
            replace(
                item,
                llm_rank=vote.rank,
                llm_confidence=vote.confidence,
                llm_tiebreak_score=_bounded_borda_score(
                    vote.rank,
                    count,
                    vote.confidence,
                    max_tiebreak_score,
                ),
            )
        )

    preferred: str | None = None
    eligible = [
        vote
        for vote in vote_items
        if vote.confidence >= minimum_preference_confidence
    ]
    if eligible:
        preferred = min(eligible, key=lambda vote: (vote.rank, vote.candidate_id)).candidate_id

    return replace(
        state,
        hypotheses=tuple(annotated),
        llm_preferred_candidate_id=preferred,
    )


def attach_normalized_transcript(
    state: TranscriptState,
    normalized_text: str,
    *,
    method: str,
) -> TranscriptState:
    """Store linguistic normalization in its own field."""

    if not method:
        raise ValueError("normalization method is required")
    return replace(
        state,
        normalized_transcript=normalized_text,
        normalization_method=method,
    )
