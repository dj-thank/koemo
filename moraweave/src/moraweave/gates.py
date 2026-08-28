from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from statistics import fmean
from typing import Mapping

from .calibration import (
    CalibrationProfile,
    ScoreRankFeatures,
    calibrate_values,
    score_rank_confidence,
)
from .contracts import CandidateEvidence, EvidenceName, GateDecision, RankedCandidate

STREAMS: tuple[EvidenceName, ...] = (
    "acoustic",
    "mora",
    "lexical",
    "preservation",
)


@dataclass(frozen=True, slots=True)
class GateConfig:
    priors: dict[EvidenceName, float]
    calibration_profiles: dict[EvidenceName, CalibrationProfile] = field(
        default_factory=dict
    )
    relisten_entropy: float = 0.56
    relisten_disagreement: float = 0.24
    relisten_margin: float = 0.18
    max_selective_risk: float = 0.33
    minimum_evidence_coverage: float = 0.58
    acceptance_posterior: float = 0.66
    grammar_honeytrap_strength: float = 0.22
    grammar_honeytrap_deadband: float = 0.08
    missing_evidence_penalty: float = 0.10
    source_diversity_bonus: float = 0.015
    posterior_temperature: float = 0.14
    stream_temperature: float = 0.20
    acoustic_family_floor: float = 0.62

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

    def __post_init__(self) -> None:
        if set(self.priors) != set(STREAMS):
            raise ValueError(f"priors must contain exactly {STREAMS}")
        if any(
            not math.isfinite(value) or value < 0 for value in self.priors.values()
        ):
            raise ValueError("gate priors must be finite and non-negative")
        if sum(self.priors.values()) <= 0:
            raise ValueError("at least one gate prior must be positive")
        if self.posterior_temperature <= 0 or self.stream_temperature <= 0:
            raise ValueError("softmax temperatures must be positive")


def _softmax(values: list[float], temperature: float) -> list[float]:
    if not values:
        return []
    maximum = max(values)
    exponentials = [
        math.exp(max(-80.0, min(80.0, (value - maximum) / temperature)))
        for value in values
    ]
    total = sum(exponentials) or 1.0
    return [value / total for value in exponentials]


def _entropy(probabilities: list[float]) -> float:
    if len(probabilities) <= 1:
        return 0.0
    raw = -sum(
        probability * math.log(probability + 1e-12)
        for probability in probabilities
    )
    return min(1.0, max(0.0, raw / math.log(len(probabilities))))


def _kl_divergence(left: list[float], right: list[float]) -> float:
    return sum(
        probability * math.log((probability + 1e-12) / (other + 1e-12))
        for probability, other in zip(left, right, strict=True)
        if probability > 0
    )


def _jensen_shannon(distributions: list[list[float]]) -> float:
    if len(distributions) <= 1:
        return 0.0
    mixture = [
        sum(distribution[index] for distribution in distributions)
        / len(distributions)
        for index in range(len(distributions[0]))
    ]
    raw = sum(
        _kl_divergence(distribution, mixture) for distribution in distributions
    ) / len(distributions)
    maximum = math.log(max(2, len(distributions[0])))
    return min(1.0, max(0.0, raw / maximum))


def _stream_reliability(values: list[float | None]) -> float:
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return 0.0
    coverage = len(finite) / len(values)
    spread = max(finite) - min(finite) if len(finite) > 1 else 0.0
    return coverage * min(1.0, 0.25 + 0.75 * spread)


def _enforce_acoustic_family_floor(
    weights: dict[EvidenceName, float],
    floor: float,
) -> dict[EvidenceName, float]:
    floor = min(1.0, max(0.0, floor))
    family = weights["acoustic"] + weights["mora"]
    if family >= floor or family <= 0:
        return weights
    language_total = weights["lexical"] + weights["preservation"]
    if language_total <= 0:
        return weights
    result = dict(weights)
    family_scale = floor / family
    language_scale = (1.0 - floor) / language_total
    result["acoustic"] *= family_scale
    result["mora"] *= family_scale
    result["lexical"] *= language_scale
    result["preservation"] *= language_scale
    return result


def _calibration_digest(
    profiles: Mapping[EvidenceName, CalibrationProfile],
) -> str:
    payload = {
        stream: profiles[stream].digest if stream in profiles else "auto-v2"
        for stream in STREAMS
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _beam_confidences(
    candidates: list[CandidateEvidence],
) -> list[float | None]:
    score_domain = [
        candidate.avg_logprob
        if candidate.avg_logprob is not None
        else candidate.sequence_score
        for candidate in candidates
    ]
    ordered_scores = sorted(
        (
            float(score)
            for score in score_domain
            if score is not None and math.isfinite(float(score))
        ),
        reverse=True,
    )
    result: list[float | None] = []
    for candidate in candidates:
        if candidate.beam_confidence is not None:
            result.append(min(1.0, max(0.0, float(candidate.beam_confidence))))
            continue
        if candidate.rank is None or candidate.hypothesis_count is None:
            result.append(None)
            continue
        score = (
            candidate.avg_logprob
            if candidate.avg_logprob is not None
            else candidate.sequence_score
        )
        margin = None
        if score is not None and ordered_scores:
            try:
                position = ordered_scores.index(float(score))
            except ValueError:
                position = -1
            if 0 <= position < len(ordered_scores) - 1:
                margin = float(score) - ordered_scores[position + 1]
            elif position == 0 and len(ordered_scores) == 1:
                margin = 1.0
        result.append(
            score_rank_confidence(
                ScoreRankFeatures(
                    rank=candidate.rank,
                    hypothesis_count=candidate.hypothesis_count,
                    avg_logprob=candidate.avg_logprob,
                    margin_to_next=margin,
                    token_count=len(candidate.token_ids) or None,
                )
            )
        )
    return result


def _candidate_source_bonus(
    candidate: CandidateEvidence,
    config: GateConfig,
) -> float:
    sources = candidate.metadata.get("sourceSupport")
    if not isinstance(sources, (list, tuple, set)):
        return 0.0
    distinct = {str(source) for source in sources if str(source)}
    return config.source_diversity_bonus * min(3, max(0, len(distinct) - 1))


def gate_candidates(
    candidates: list[CandidateEvidence],
    config: GateConfig | None = None,
) -> list[RankedCandidate]:
    """Fuse heterogeneous evidence with calibration, uncertainty and abstention."""

    if not candidates:
        raise ValueError("at least one candidate is required")
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise ValueError("candidate IDs must be unique")
    config = config or GateConfig.default()
    calibrated: dict[EvidenceName, list[float | None]] = {
        stream: calibrate_values(
            [candidate.score(stream) for candidate in candidates],
            profile=config.calibration_profiles.get(stream),
            stream_name=stream,
        )
        for stream in STREAMS
    }
    beam_confidences = _beam_confidences(candidates)
    for index, beam_confidence in enumerate(beam_confidences):
        if beam_confidence is None:
            continue
        acoustic = calibrated["acoustic"][index]
        calibrated["acoustic"][index] = (
            beam_confidence
            if acoustic is None
            else 0.75 * acoustic + 0.25 * beam_confidence
        )
    reliabilities = {
        stream: _stream_reliability(calibrated[stream]) for stream in STREAMS
    }
    raw_weights = {
        stream: config.priors[stream] * (0.25 + reliabilities[stream])
        for stream in STREAMS
    }
    total = sum(raw_weights.values()) or 1.0
    weights = {stream: raw_weights[stream] / total for stream in STREAMS}
    weights = _enforce_acoustic_family_floor(weights, config.acoustic_family_floor)
    stream_distributions: list[list[float]] = []
    for stream in STREAMS:
        values = calibrated[stream]
        if not any(value is not None for value in values):
            continue
        logits = [-8.0 if value is None else float(value) for value in values]
        stream_distributions.append(_softmax(logits, config.stream_temperature))
    disagreement = _jensen_shannon(stream_distributions)
    preliminary = []
    for index, candidate in enumerate(candidates):
        parts = {
            stream: float(calibrated[stream][index])
            if calibrated[stream][index] is not None
            else 0.0
            for stream in STREAMS
        }
        present_weight = sum(
            weights[stream]
            for stream in STREAMS
            if calibrated[stream][index] is not None
        )
        missing_penalty = (1.0 - present_weight) * config.missing_evidence_penalty
        source_bonus = _candidate_source_bonus(candidate, config)
        score = sum(weights[stream] * parts[stream] for stream in STREAMS)
        score -= missing_penalty
        score += source_bonus
        acoustic_weights = weights["acoustic"] + weights["mora"]
        acoustic_support = (
            (weights["acoustic"] * parts["acoustic"] + weights["mora"] * parts["mora"])
            / acoustic_weights
            if acoustic_weights > 0
            else 0.0
        )
        grammar_honeytrap = 0.0
        if candidate.teacher is not None:
            teacher_probability = min(1.0, max(0.0, float(candidate.teacher)))
            unsupported_preference = max(
                0.0,
                teacher_probability - acoustic_support - config.grammar_honeytrap_deadband,
            )
            grammar_honeytrap = config.grammar_honeytrap_strength * unsupported_preference
            score -= grammar_honeytrap
        preliminary.append(
            (candidate, score, parts, grammar_honeytrap, missing_penalty, source_bonus, present_weight)
        )
    posterior = _softmax([item[1] for item in preliminary], config.posterior_temperature)
    order = sorted(
        range(len(preliminary)),
        key=lambda index: (-posterior[index], -preliminary[index][1], preliminary[index][0].candidate_id),
    )
    top_index = order[0]
    top_posterior = posterior[top_index]
    second_posterior = posterior[order[1]] if len(order) > 1 else 0.0
    margin = top_posterior - second_posterior
    entropy = _entropy(posterior)
    evidence_coverage = preliminary[top_index][6]
    selective_risk = min(
        1.0,
        max(
            0.0,
            0.60 * (1.0 - top_posterior)
            + 0.22 * disagreement
            + 0.18 * (1.0 - evidence_coverage),
        ),
    )
    needs_relisten = (
        entropy >= config.relisten_entropy
        or disagreement >= config.relisten_disagreement
        or margin <= config.relisten_margin
        or selective_risk >= config.max_selective_risk
        or evidence_coverage < config.minimum_evidence_coverage
    )
    abstain = (
        top_posterior < config.acceptance_posterior
        and (
            selective_risk >= config.max_selective_risk
            or evidence_coverage < config.minimum_evidence_coverage
            or disagreement >= config.relisten_disagreement
        )
    )
    reasons: list[str] = []
    if entropy >= config.relisten_entropy:
        reasons.append("high-candidate-entropy")
    if disagreement >= config.relisten_disagreement:
        reasons.append("evidence-stream-disagreement")
    if margin <= config.relisten_margin:
        reasons.append("small-posterior-margin")
    if selective_risk >= config.max_selective_risk:
        reasons.append("high-selective-risk")
    if evidence_coverage < config.minimum_evidence_coverage:
        reasons.append("low-evidence-coverage")
    if abstain:
        reasons.append("provisional-observation")
    posterior_map = {
        candidate.candidate_id: probability
        for candidate, probability in zip(candidates, posterior, strict=True)
    }
    gate = GateDecision(
        weights=weights,
        disagreement=disagreement,
        entropy=entropy,
        needs_relisten=needs_relisten,
        reasons=tuple(reasons),
        posterior=posterior_map,
        evidence_coverage=evidence_coverage,
        selective_risk=selective_risk,
        abstain=abstain,
        calibration_digest=_calibration_digest(config.calibration_profiles),
        uncertainty={
            "aleatoric": entropy,
            "epistemic": disagreement,
            "missingEvidence": 1.0 - evidence_coverage,
            "posteriorMargin": margin,
        },
    )
    ranked = [
        RankedCandidate(
            candidate=candidate,
            final_score=score,
            normalized_scores=parts,
            gate=gate,
            grammar_honeytrap_penalty=honeytrap,
            posterior=probability,
            missing_evidence_penalty=missing_penalty,
            source_diversity_bonus=source_bonus,
        )
        for (
            candidate,
            score,
            parts,
            honeytrap,
            missing_penalty,
            source_bonus,
            _coverage,
        ), probability in zip(preliminary, posterior, strict=True)
    ]
    return sorted(
        ranked,
        key=lambda item: (-item.posterior, -item.final_score, item.candidate.candidate_id),
    )


def evidence_summary(
    ranked: list[RankedCandidate],
) -> dict[str, float | bool | list[str] | str | dict[str, float]]:
    if not ranked:
        raise ValueError("ranked candidates are required")
    gate = ranked[0].gate
    return {
        "entropy": gate.entropy,
        "disagreement": gate.disagreement,
        "evidenceCoverage": gate.evidence_coverage,
        "selectiveRisk": gate.selective_risk,
        "abstain": gate.abstain,
        "needsRelisten": gate.needs_relisten,
        "reasons": list(gate.reasons),
        "topScore": ranked[0].final_score,
        "topPosterior": ranked[0].posterior,
        "meanScore": fmean(item.final_score for item in ranked),
        "calibrationDigest": gate.calibration_digest or "",
        "uncertainty": dict(gate.uncertainty),
    }
