"""Mora-aware minimum-Bayes-risk selection for Whisper N-best lists.

The decoder always selects an existing acoustic hypothesis. It never creates or
rewrites transcript text, preserving the ``observedTranscript`` invariant while
using agreement across the complete N-best list to avoid brittle one-best picks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import exp, isfinite, log
from typing import Callable, Iterable, Mapping, Sequence

from .calibration import normalized_entropy, softmax
from .mora import mora_units_from_reading, normalize_kana, split_mora
from .schema import RankedHypothesis, TranscriptHypothesis, TranscriptState, UnitKind
from .scoring import ScoreWeights, rank_acoustic_hypotheses

ReadingResolver = Callable[[str], str]
_EPSILON = 1e-12


class DecisionStatus(str, Enum):
    """Operational disposition of a consensus decision."""

    ACCEPT = "accept"
    REVIEW = "review"
    NO_SPEECH = "no_speech"


@dataclass(frozen=True, slots=True)
class MoraEditCosts:
    """Configurable edit costs; defaults preserve learner errors equally."""

    insertion: float = 1.0
    deletion: float = 1.0
    substitution: float = 1.0
    confusion_costs: Mapping[tuple[str, str], float] = field(
        default_factory=dict, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        for value, name in (
            (self.insertion, "insertion"),
            (self.deletion, "deletion"),
            (self.substitution, "substitution"),
        ):
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} cost must be finite and non-negative")
        for key, value in self.confusion_costs.items():
            if len(key) != 2 or not all(isinstance(part, str) and part for part in key):
                raise ValueError("confusion cost keys must be two non-empty strings")
            if not isfinite(value) or value < 0.0:
                raise ValueError("confusion costs must be finite and non-negative")

    def substitution_cost(self, left: str, right: str) -> float:
        if left == right:
            return 0.0
        direct = self.confusion_costs.get((left, right))
        if direct is not None:
            return direct
        reverse = self.confusion_costs.get((right, left))
        return self.substitution if reverse is None else reverse


@dataclass(frozen=True, slots=True)
class ConsensusConfig:
    """Selection and abstention thresholds for mora-aware MBR decoding."""

    posterior_temperature: float = 1.0
    acoustic_tiebreak_weight: float = 0.03
    min_selected_posterior: float = 0.30
    min_decision_margin: float = 0.02
    max_normalized_entropy: float = 0.88
    max_bayes_risk: float = 0.52
    max_no_speech_probability: float = 0.65
    average_duplicate_logits: bool = True

    def __post_init__(self) -> None:
        if not isfinite(self.posterior_temperature) or self.posterior_temperature <= 0.0:
            raise ValueError("posterior_temperature must be finite and positive")
        if not isfinite(self.acoustic_tiebreak_weight) or self.acoustic_tiebreak_weight < 0.0:
            raise ValueError("acoustic_tiebreak_weight must be finite and non-negative")
        for value, name in (
            (self.min_selected_posterior, "min_selected_posterior"),
            (self.max_normalized_entropy, "max_normalized_entropy"),
            (self.max_bayes_risk, "max_bayes_risk"),
            (self.max_no_speech_probability, "max_no_speech_probability"),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not isfinite(self.min_decision_margin) or self.min_decision_margin < 0.0:
            raise ValueError("min_decision_margin must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ConsensusCandidate:
    candidate_id: str
    duplicate_candidate_ids: tuple[str, ...]
    units: tuple[str, ...]
    acoustic_score: float
    posterior: float
    bayes_risk: float
    consensus_score: float

    def to_dict(self) -> dict[str, object]:
        return {
            "candidateId": self.candidate_id,
            "duplicateCandidateIds": list(self.duplicate_candidate_ids),
            "units": list(self.units),
            "acousticScore": self.acoustic_score,
            "posterior": self.posterior,
            "bayesRisk": self.bayes_risk,
            "consensusScore": self.consensus_score,
        }


@dataclass(frozen=True, slots=True)
class ConsensusDecision:
    selected_candidate_id: str
    acoustic_top_candidate_id: str
    status: DecisionStatus
    reasons: tuple[str, ...]
    selected_posterior: float
    posterior_margin: float
    decision_margin: float
    entropy: float
    selected_bayes_risk: float
    candidates: tuple[ConsensusCandidate, ...]

    @property
    def overrode_acoustic_top(self) -> bool:
        return self.selected_candidate_id != self.acoustic_top_candidate_id

    def to_dict(self) -> dict[str, object]:
        return {
            "selectedCandidateId": self.selected_candidate_id,
            "acousticTopCandidateId": self.acoustic_top_candidate_id,
            "overrodeAcousticTop": self.overrode_acoustic_top,
            "status": self.status.value,
            "reasons": list(self.reasons),
            "selectedPosterior": self.selected_posterior,
            "posteriorMargin": self.posterior_margin,
            "decisionMargin": self.decision_margin,
            "normalizedEntropy": self.entropy,
            "selectedBayesRisk": self.selected_bayes_risk,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def mora_edit_distance(
    left: Sequence[str],
    right: Sequence[str],
    *,
    costs: MoraEditCosts = MoraEditCosts(),
) -> float:
    """Weighted Levenshtein distance over mora or fallback grapheme units."""

    previous = [index * costs.insertion for index in range(len(right) + 1)]
    for left_index, left_unit in enumerate(left, start=1):
        current = [left_index * costs.deletion]
        for right_index, right_unit in enumerate(right, start=1):
            current.append(
                min(
                    previous[right_index] + costs.deletion,
                    current[right_index - 1] + costs.insertion,
                    previous[right_index - 1]
                    + costs.substitution_cost(left_unit, right_unit),
                )
            )
        previous = current
    return previous[-1]


def normalized_mora_distance(
    left: Sequence[str],
    right: Sequence[str],
    *,
    costs: MoraEditCosts = MoraEditCosts(),
) -> float:
    denominator = max(len(left), len(right), 1)
    maximum_unit_cost = max(costs.insertion, costs.deletion, costs.substitution, 1.0)
    distance = mora_edit_distance(left, right, costs=costs)
    return min(1.0, distance / (denominator * maximum_unit_cost))


def _candidate_units(
    hypothesis: TranscriptHypothesis,
    reading_resolver: ReadingResolver | None,
) -> tuple[str, ...]:
    if reading_resolver is not None:
        reading = reading_resolver(hypothesis.text)
        if not isinstance(reading, str) or not reading.strip():
            raise ValueError(
                f"reading resolver returned no reading for {hypothesis.candidate_id}"
            )
        moras = split_mora(reading)
        if moras:
            return moras

    if hypothesis.mora_units:
        labels = tuple(
            unit.mora
            if unit.kind is UnitKind.MORA
            else normalize_kana(unit.reading or unit.surface)
            for unit in hypothesis.mora_units
            if unit.kind is not UnitKind.BOUNDARY
            and (unit.mora or unit.reading or unit.surface)
        )
        if labels:
            return labels

    # Without a morphological reading resolver, kana are merged into moras and
    # kanji/Latin characters remain visible as individual fallback units.
    labels: list[str] = []
    for unit in mora_units_from_reading(hypothesis.text, keep_boundaries=False):
        if unit.kind is UnitKind.MORA:
            labels.append(unit.mora)
        elif unit.kind is not UnitKind.BOUNDARY:
            value = normalize_kana(unit.reading or unit.surface)
            if value:
                labels.append(value)
    if labels:
        return tuple(labels)

    normalized = normalize_kana(hypothesis.text)
    fallback = tuple(character for character in normalized if not character.isspace())
    return fallback or ("<EMPTY>",)


def _log_mean_exp(values: Sequence[float]) -> float:
    maximum = max(values)
    return maximum + log(sum(exp(value - maximum) for value in values)) - log(len(values))


def decide_mora_consensus(
    ranked_hypotheses: Iterable[RankedHypothesis],
    *,
    config: ConsensusConfig = ConsensusConfig(),
    costs: MoraEditCosts = MoraEditCosts(),
    reading_resolver: ReadingResolver | None = None,
) -> ConsensusDecision:
    """Apply candidate-preserving MBR decoding to acoustically ranked N-best."""

    ranked = tuple(ranked_hypotheses)
    if not ranked:
        raise ValueError("at least one ranked hypothesis is required")
    ids = [item.hypothesis.candidate_id for item in ranked]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate IDs must be unique")

    grouped: dict[tuple[str, ...], list[RankedHypothesis]] = {}
    for item in ranked:
        key = _candidate_units(item.hypothesis, reading_resolver)
        grouped.setdefault(key, []).append(item)

    groups: list[tuple[tuple[str, ...], RankedHypothesis, tuple[str, ...], float]] = []
    for units, members in grouped.items():
        representative = min(
            members,
            key=lambda item: (-item.acoustic_score, item.hypothesis.candidate_id),
        )
        scaled_scores = tuple(
            item.acoustic_score / config.posterior_temperature for item in members
        )
        if config.average_duplicate_logits:
            group_logit = _log_mean_exp(scaled_scores)
        else:
            maximum = max(scaled_scores)
            group_logit = maximum + log(sum(exp(value - maximum) for value in scaled_scores))
        member_ids = tuple(
            item.hypothesis.candidate_id
            for item in sorted(
                members,
                key=lambda item: (-item.acoustic_score, item.hypothesis.candidate_id),
            )
        )
        groups.append((units, representative, member_ids, group_logit))

    groups.sort(key=lambda item: (-item[3], item[1].hypothesis.candidate_id))
    posteriors = softmax(tuple(group[3] for group in groups))

    candidates: list[ConsensusCandidate] = []
    for index, (units, representative, member_ids, _) in enumerate(groups):
        risk = sum(
            posteriors[other_index]
            * normalized_mora_distance(units, other_units, costs=costs)
            for other_index, (other_units, _, _, _) in enumerate(groups)
        )
        posterior = posteriors[index]
        score = -risk + config.acoustic_tiebreak_weight * log(max(posterior, _EPSILON))
        candidates.append(
            ConsensusCandidate(
                candidate_id=representative.hypothesis.candidate_id,
                duplicate_candidate_ids=member_ids,
                units=units,
                acoustic_score=representative.acoustic_score,
                posterior=posterior,
                bayes_risk=risk,
                consensus_score=score,
            )
        )

    candidates.sort(key=lambda item: (-item.consensus_score, item.candidate_id))
    selected = candidates[0]
    second_score = (
        candidates[1].consensus_score
        if len(candidates) > 1
        else selected.consensus_score - 1.0
    )
    decision_margin = selected.consensus_score - second_score
    other_posteriors = [
        candidate.posterior
        for candidate in candidates
        if candidate.candidate_id != selected.candidate_id
    ]
    posterior_margin = selected.posterior - (
        max(other_posteriors) if other_posteriors else 0.0
    )
    entropy = normalized_entropy(tuple(candidate.posterior for candidate in candidates))

    by_id = {item.hypothesis.candidate_id: item for item in ranked}
    selected_hypothesis = by_id[selected.candidate_id].hypothesis
    no_speech = selected_hypothesis.features.no_speech_probability
    reasons: list[str] = []
    status = DecisionStatus.ACCEPT
    if no_speech is not None and no_speech >= config.max_no_speech_probability:
        status = DecisionStatus.NO_SPEECH
        reasons.append("high_no_speech_probability")
    else:
        if selected.posterior < config.min_selected_posterior:
            reasons.append("low_selected_posterior")
        if decision_margin < config.min_decision_margin:
            reasons.append("small_consensus_margin")
        if entropy > config.max_normalized_entropy:
            reasons.append("high_nbest_entropy")
        if selected.bayes_risk > config.max_bayes_risk:
            reasons.append("high_bayes_risk")
        if reasons:
            status = DecisionStatus.REVIEW

    return ConsensusDecision(
        selected_candidate_id=selected.candidate_id,
        acoustic_top_candidate_id=min(
            ranked, key=lambda item: item.rank
        ).hypothesis.candidate_id,
        status=status,
        reasons=tuple(reasons),
        selected_posterior=selected.posterior,
        posterior_margin=posterior_margin,
        decision_margin=decision_margin,
        entropy=entropy,
        selected_bayes_risk=selected.bayes_risk,
        candidates=tuple(candidates),
    )


def select_consensus_observed_transcript(
    hypotheses: Iterable[TranscriptHypothesis],
    *,
    weights: ScoreWeights = ScoreWeights(),
    config: ConsensusConfig = ConsensusConfig(),
    costs: MoraEditCosts = MoraEditCosts(),
    reading_resolver: ReadingResolver | None = None,
) -> tuple[TranscriptState, ConsensusDecision]:
    """Rank acoustically, run MBR consensus, and freeze an existing candidate."""

    ranked = rank_acoustic_hypotheses(hypotheses, weights=weights)
    decision = decide_mora_consensus(
        ranked,
        config=config,
        costs=costs,
        reading_resolver=reading_resolver,
    )
    winner = next(
        item.hypothesis
        for item in ranked
        if item.hypothesis.candidate_id == decision.selected_candidate_id
    )
    state = TranscriptState(
        observed_transcript=winner.text,
        observed_candidate_id=winner.candidate_id,
        hypotheses=ranked,
        mora_units=winner.mora_units,
    )
    return state, decision
