from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean, pstdev

from .contracts import CandidateEvidence, EvidenceName, GateDecision, RankedCandidate

STREAMS: tuple[EvidenceName, ...] = ("acoustic", "mora", "lexical", "preservation")


@dataclass(frozen=True, slots=True)
class GateConfig:
    priors: dict[EvidenceName, float]
    relisten_entropy: float = 0.58
    relisten_disagreement: float = 0.38
    relisten_margin: float = 0.12
    grammar_honeytrap_strength: float = 0.22

    @classmethod
    def default(cls) -> "GateConfig":
        return cls(
            priors={
                "acoustic": 0.48,
                "mora": 0.24,
                "lexical": 0.14,
                "preservation": 0.14,
            }
        )


def _minmax(values: list[float | None]) -> list[float | None]:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return [None] * len(values)
    low, high = min(finite), max(finite)
    if math.isclose(low, high):
        return [1.0 if value is not None else None for value in values]
    return [None if value is None else (float(value) - low) / (high - low) for value in values]


def _entropy(probabilities: list[float]) -> float:
    if len(probabilities) <= 1:
        return 0.0
    total = sum(probabilities)
    if total <= 0:
        return 1.0
    normalized = [value / total for value in probabilities]
    raw = -sum(value * math.log(value + 1e-12) for value in normalized)
    return raw / math.log(len(normalized))


def _softmax(values: list[float], temperature: float = 0.18) -> list[float]:
    maximum = max(values)
    exps = [math.exp((value - maximum) / temperature) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def _reliability(stream_values: list[float | None]) -> float:
    finite = [value for value in stream_values if value is not None]
    if not finite:
        return 0.0
    coverage = len(finite) / len(stream_values)
    separation = pstdev(finite) if len(finite) > 1 else 0.0
    return coverage * min(1.0, 0.35 + separation * 1.8)


def gate_candidates(
    candidates: list[CandidateEvidence],
    config: GateConfig | None = None,
) -> list[RankedCandidate]:
    if not candidates:
        raise ValueError("at least one candidate is required")
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise ValueError("candidate IDs must be unique")

    config = config or GateConfig.default()
    normalized_by_stream: dict[EvidenceName, list[float | None]] = {
        stream: _minmax([candidate.score(stream) for candidate in candidates])
        for stream in STREAMS
    }
    reliability = {
        stream: _reliability(normalized_by_stream[stream]) for stream in STREAMS
    }
    raw_weights = {
        stream: config.priors[stream] * (0.2 + reliability[stream]) for stream in STREAMS
    }
    weight_total = sum(raw_weights.values()) or 1.0
    weights = {stream: value / weight_total for stream, value in raw_weights.items()}

    stream_winners: list[int] = []
    for stream in STREAMS:
        values = normalized_by_stream[stream]
        available = [(index, value) for index, value in enumerate(values) if value is not None]
        if available:
            stream_winners.append(max(available, key=lambda item: item[1])[0])
    winner_disagreement = (
        0.0 if not stream_winners else 1.0 - max(stream_winners.count(index) for index in set(stream_winners)) / len(stream_winners)
    )

    preliminary: list[tuple[CandidateEvidence, float, dict[str, float], float]] = []
    for index, candidate in enumerate(candidates):
        parts = {
            stream: float(normalized_by_stream[stream][index] or 0.0) for stream in STREAMS
        }
        score = sum(weights[stream] * parts[stream] for stream in STREAMS)

        teacher = candidate.teacher
        acoustic_support = (parts["acoustic"] + parts["mora"]) / 2
        grammar_honeytrap = 0.0
        if teacher is not None and math.isfinite(teacher):
            clean_preference = max(0.0, float(teacher) - acoustic_support)
            grammar_honeytrap = config.grammar_honeytrap_strength * clean_preference
            score -= grammar_honeytrap
        preliminary.append((candidate, score, parts, grammar_honeytrap))

    probabilities = _softmax([item[1] for item in preliminary])
    entropy = _entropy(probabilities)
    order = sorted(range(len(preliminary)), key=lambda index: preliminary[index][1], reverse=True)
    margin = 1.0 if len(order) == 1 else preliminary[order[0]][1] - preliminary[order[1]][1]
    needs_relisten = (
        entropy >= config.relisten_entropy
        or winner_disagreement >= config.relisten_disagreement
        or margin <= config.relisten_margin
    )
    reasons: list[str] = []
    if entropy >= config.relisten_entropy:
        reasons.append("high-candidate-entropy")
    if winner_disagreement >= config.relisten_disagreement:
        reasons.append("evidence-stream-disagreement")
    if margin <= config.relisten_margin:
        reasons.append("small-top-two-margin")

    gate = GateDecision(
        weights=weights,
        disagreement=winner_disagreement,
        entropy=entropy,
        needs_relisten=needs_relisten,
        reasons=tuple(reasons),
    )
    ranked = [
        RankedCandidate(
            candidate=candidate,
            final_score=score,
            normalized_scores=parts,
            gate=gate,
            grammar_honeytrap_penalty=penalty,
        )
        for candidate, score, parts, penalty in preliminary
    ]
    return sorted(
        ranked,
        key=lambda item: (-item.final_score, item.candidate.candidate_id),
    )


def evidence_summary(ranked: list[RankedCandidate]) -> dict[str, float | bool | list[str]]:
    if not ranked:
        raise ValueError("ranked candidates are required")
    gate = ranked[0].gate
    return {
        "entropy": gate.entropy,
        "disagreement": gate.disagreement,
        "needsRelisten": gate.needs_relisten,
        "reasons": list(gate.reasons),
        "topScore": ranked[0].final_score,
        "meanScore": fmean(item.final_score for item in ranked),
    }
