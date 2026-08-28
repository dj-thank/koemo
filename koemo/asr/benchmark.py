"""Reproducible Japanese ASR benchmark, oracle analysis, and regression gates.

This module contains no model runtime. It evaluates already produced predictions,
which keeps benchmark reports deterministic and lets the same manifest compare
Whisper, mora-CTC, forced-alignment, and LLM-reranked systems.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from random import Random
from statistics import mean
from typing import Iterable, Mapping, Sequence

from .evaluation import (
    ErrorRate,
    LearnerErrorPreservation,
    character_error_rate,
    evaluate_japanese_asr,
    learner_error_preservation,
    mora_error_rate,
)


class BenchmarkSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    CALIBRATION = "calibration"
    TEST = "test"
    CHALLENGE = "challenge"


_VALID_STATUSES = frozenset(("accept", "review", "no_speech"))


@dataclass(frozen=True, slots=True)
class BenchmarkUtterance:
    utterance_id: str
    speaker_id: str
    split: BenchmarkSplit
    reference_text: str
    is_speech: bool = True
    reference_reading: str | None = None
    target_reading: str | None = None
    observed_reading: str | None = None
    audio_sha256: str | None = None
    groups: Mapping[str, str] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if not self.utterance_id:
            raise ValueError("utterance_id is required")
        if not self.speaker_id:
            raise ValueError("speaker_id is required")
        if not isinstance(self.split, BenchmarkSplit):
            raise TypeError("split must be a BenchmarkSplit")
        if not isinstance(self.reference_text, str):
            raise TypeError("reference_text must be a string")
        if not isinstance(self.is_speech, bool):
            raise TypeError("is_speech must be a bool")
        if self.is_speech and not self.reference_text.strip():
            raise ValueError("speech utterances require reference_text")
        if not self.is_speech:
            if self.reference_text.strip():
                raise ValueError("non-speech utterances must use empty reference_text")
            for value, name in (
                (self.reference_reading, "reference_reading"),
                (self.target_reading, "target_reading"),
                (self.observed_reading, "observed_reading"),
            ):
                if value is not None and value.strip():
                    raise ValueError(f"non-speech utterances cannot define {name}")
        if self.audio_sha256 is not None:
            digest = self.audio_sha256.lower()
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("audio_sha256 must be a 64-character hexadecimal digest")
        if any(not key or not value for key, value in self.groups.items()):
            raise ValueError("group keys and values must be non-empty")


@dataclass(frozen=True, slots=True)
class NBestCandidate:
    candidate_id: str
    text: str
    reading: str | None = None
    score: float | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id is required")
        if not isinstance(self.text, str):
            raise TypeError("candidate text must be a string")
        if self.reading is not None and not isinstance(self.reading, str):
            raise TypeError("candidate reading must be a string or null")
        if self.score is not None and not isfinite(self.score):
            raise ValueError("candidate score must be finite")


@dataclass(frozen=True, slots=True)
class SystemPrediction:
    utterance_id: str
    system_id: str
    text: str
    reading: str | None = None
    status: str = "accept"
    latency_ms: float | None = None
    candidates: tuple[NBestCandidate, ...] = ()

    def __post_init__(self) -> None:
        if not self.utterance_id:
            raise ValueError("utterance_id is required")
        if not self.system_id:
            raise ValueError("system_id is required")
        if not isinstance(self.text, str):
            raise TypeError("prediction text must be a string")
        if self.reading is not None and not isinstance(self.reading, str):
            raise TypeError("prediction reading must be a string or null")
        if self.status not in _VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(_VALID_STATUSES)}")
        if self.status == "no_speech":
            if self.text.strip():
                raise ValueError("no_speech predictions must use empty text")
            if self.reading is not None and self.reading.strip():
                raise ValueError("no_speech predictions must use empty reading")
        if self.latency_ms is not None:
            if not isfinite(self.latency_ms) or self.latency_ms < 0.0:
                raise ValueError("latency_ms must be finite and non-negative")
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("N-best candidate IDs must be unique per prediction")


@dataclass(frozen=True, slots=True)
class ManifestIssue:
    code: str
    message: str
    utterance_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class UtteranceEvaluation:
    utterance_id: str
    speaker_id: str
    split: BenchmarkSplit
    system_id: str
    status: str
    reference_is_speech: bool
    predicted_is_speech: bool
    cer: ErrorRate
    kana_cer: ErrorRate | None
    mora_error_rate: ErrorRate | None
    learner_preservation: LearnerErrorPreservation | None
    oracle_cer: ErrorRate | None
    oracle_mora_error_rate: ErrorRate | None
    oracle_cer_candidate_id: str | None
    oracle_mora_candidate_id: str | None
    latency_ms: float | None


@dataclass(frozen=True, slots=True)
class AggregateSystemMetrics:
    system_id: str
    utterance_count: int
    cer: ErrorRate
    kana_cer: ErrorRate | None
    mora_error_rate: ErrorRate | None
    oracle_cer: ErrorRate | None
    oracle_mora_error_rate: ErrorRate | None
    accepted_cer: ErrorRate | None
    accepted_mora_error_rate: ErrorRate | None
    kana_evaluated: int
    mora_evaluated: int
    learner_evaluated: int
    oracle_cer_evaluated: int
    oracle_mora_evaluated: int
    accepted_speech_count: int
    mean_learner_preservation: float | None
    normalized_to_target_rate: float | None
    accept_count: int
    review_count: int
    no_speech_count: int
    reference_speech_count: int
    reference_no_speech_count: int
    predicted_speech_count: int
    true_speech_count: int
    missed_speech_count: int
    true_no_speech_count: int
    hallucinated_speech_count: int
    speech_accept_rate: float | None
    no_speech_precision: float | None
    no_speech_recall: float | None
    no_speech_f1: float | None
    hallucination_on_silence_rate: float | None
    missed_speech_rate: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "systemId": self.system_id,
            "utteranceCount": self.utterance_count,
            "cer": self.cer.to_dict(),
            "kanaCer": None if self.kana_cer is None else self.kana_cer.to_dict(),
            "moraErrorRate": (
                None
                if self.mora_error_rate is None
                else self.mora_error_rate.to_dict()
            ),
            "oracleCer": (
                None if self.oracle_cer is None else self.oracle_cer.to_dict()
            ),
            "oracleMoraErrorRate": (
                None
                if self.oracle_mora_error_rate is None
                else self.oracle_mora_error_rate.to_dict()
            ),
            "selective": {
                "acceptedSpeech": self.accepted_speech_count,
                "speechAcceptRate": self.speech_accept_rate,
                "acceptedCer": (
                    None if self.accepted_cer is None else self.accepted_cer.to_dict()
                ),
                "acceptedMoraErrorRate": (
                    None
                    if self.accepted_mora_error_rate is None
                    else self.accepted_mora_error_rate.to_dict()
                ),
            },
            "coverage": {
                "kana": self.kana_evaluated,
                "mora": self.mora_evaluated,
                "learner": self.learner_evaluated,
                "oracleCer": self.oracle_cer_evaluated,
                "oracleMora": self.oracle_mora_evaluated,
            },
            "meanLearnerPreservation": self.mean_learner_preservation,
            "normalizedToTargetRate": self.normalized_to_target_rate,
            "statusCounts": {
                "accept": self.accept_count,
                "review": self.review_count,
                "noSpeech": self.no_speech_count,
            },
            "speechDetection": {
                "referenceSpeech": self.reference_speech_count,
                "referenceNoSpeech": self.reference_no_speech_count,
                "predictedSpeech": self.predicted_speech_count,
                "trueSpeech": self.true_speech_count,
                "missedSpeech": self.missed_speech_count,
                "trueNoSpeech": self.true_no_speech_count,
                "hallucinatedSpeech": self.hallucinated_speech_count,
                "noSpeechPrecision": self.no_speech_precision,
                "noSpeechRecall": self.no_speech_recall,
                "noSpeechF1": self.no_speech_f1,
                "hallucinationOnSilenceRate": self.hallucination_on_silence_rate,
                "missedSpeechRate": self.missed_speech_rate,
            },
            "latencyMs": {
                "p50": self.latency_p50_ms,
                "p95": self.latency_p95_ms,
            },
        }


@dataclass(frozen=True, slots=True)
class BootstrapComparison:
    metric: str
    baseline_rate: float
    candidate_rate: float
    delta: float
    confidence_level: float
    lower_bound: float
    upper_bound: float
    improvement_probability: float
    samples: int
    cluster_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "baselineRate": self.baseline_rate,
            "candidateRate": self.candidate_rate,
            "delta": self.delta,
            "confidenceLevel": self.confidence_level,
            "lowerBound": self.lower_bound,
            "upperBound": self.upper_bound,
            "improvementProbability": self.improvement_probability,
            "samples": self.samples,
            "clusterCount": self.cluster_count,
        }


@dataclass(frozen=True, slots=True)
class AccuracyGateConfig:
    max_cer_regression: float = 0.0
    max_mora_regression: float = 0.0
    min_learner_preservation_delta: float = 0.0
    max_normalized_to_target_increase: float = 0.0
    max_hallucination_on_silence_increase: float = 0.0
    max_missed_speech_rate_increase: float = 0.0
    max_speech_accept_rate_decrease: float = 0.0
    require_cer_confidence_improvement: bool = False
    require_mora_confidence_improvement: bool = False

    def __post_init__(self) -> None:
        for value, name in (
            (self.max_cer_regression, "max_cer_regression"),
            (self.max_mora_regression, "max_mora_regression"),
            (
                self.max_normalized_to_target_increase,
                "max_normalized_to_target_increase",
            ),
            (
                self.max_hallucination_on_silence_increase,
                "max_hallucination_on_silence_increase",
            ),
            (
                self.max_missed_speech_rate_increase,
                "max_missed_speech_rate_increase",
            ),
            (
                self.max_speech_accept_rate_decrease,
                "max_speech_accept_rate_decrease",
            ),
        ):
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not isfinite(self.min_learner_preservation_delta):
            raise ValueError("min_learner_preservation_delta must be finite")


@dataclass(frozen=True, slots=True)
class AccuracyGateResult:
    passed: bool
    reasons: tuple[str, ...]
    deltas: Mapping[str, float | None] = field(
        default_factory=dict, compare=False, hash=False
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "deltas": dict(self.deltas),
        }


def validate_manifest(
    utterances: Iterable[BenchmarkUtterance],
    *,
    require_speaker_disjoint: bool = True,
    require_audio_disjoint: bool = True,
) -> tuple[ManifestIssue, ...]:
    items = tuple(utterances)
    issues: list[ManifestIssue] = []

    by_id: dict[str, list[BenchmarkUtterance]] = {}
    by_speaker: dict[str, list[BenchmarkUtterance]] = {}
    by_audio: dict[str, list[BenchmarkUtterance]] = {}
    for item in items:
        by_id.setdefault(item.utterance_id, []).append(item)
        by_speaker.setdefault(item.speaker_id, []).append(item)
        if item.audio_sha256 is not None:
            by_audio.setdefault(item.audio_sha256.lower(), []).append(item)

    for utterance_id, duplicates in by_id.items():
        if len(duplicates) > 1:
            issues.append(
                ManifestIssue(
                    code="duplicate_utterance_id",
                    message=f"utterance ID {utterance_id!r} appears more than once",
                    utterance_ids=tuple(item.utterance_id for item in duplicates),
                )
            )

    if require_speaker_disjoint:
        for speaker_id, speaker_items in by_speaker.items():
            splits = {item.split for item in speaker_items}
            if len(splits) > 1:
                issues.append(
                    ManifestIssue(
                        code="speaker_split_leakage",
                        message=(
                            f"speaker {speaker_id!r} appears in multiple splits: "
                            f"{sorted(split.value for split in splits)}"
                        ),
                        utterance_ids=tuple(item.utterance_id for item in speaker_items),
                    )
                )

    if require_audio_disjoint:
        for digest, audio_items in by_audio.items():
            splits = {item.split for item in audio_items}
            if len(audio_items) > 1 and len(splits) > 1:
                issues.append(
                    ManifestIssue(
                        code="audio_split_leakage",
                        message=(
                            f"audio SHA-256 {digest} appears in multiple splits: "
                            f"{sorted(split.value for split in splits)}"
                        ),
                        utterance_ids=tuple(item.utterance_id for item in audio_items),
                    )
                )
    return tuple(issues)


def assert_manifest_integrity(
    utterances: Iterable[BenchmarkUtterance],
    *,
    require_speaker_disjoint: bool = True,
    require_audio_disjoint: bool = True,
) -> None:
    issues = validate_manifest(
        utterances,
        require_speaker_disjoint=require_speaker_disjoint,
        require_audio_disjoint=require_audio_disjoint,
    )
    if issues:
        summary = "; ".join(f"{issue.code}: {issue.message}" for issue in issues)
        raise ValueError(summary)


def _best_error_candidate(
    candidates: Sequence[NBestCandidate],
    *,
    reference_text: str | None = None,
    reference_reading: str | None = None,
) -> tuple[ErrorRate | None, str | None]:
    scored: list[tuple[ErrorRate, str]] = []
    for candidate in candidates:
        if reference_reading is not None:
            if candidate.reading is None:
                continue
            result = mora_error_rate(reference_reading, candidate.reading)
        elif reference_text is not None:
            result = character_error_rate(reference_text, candidate.text)
        else:
            raise ValueError("a reference must be supplied")
        scored.append((result, candidate.candidate_id))
    if not scored:
        return None, None
    best = min(
        scored,
        key=lambda item: (
            item[0].errors,
            item[0].rate,
            item[1],
        ),
    )
    return best


def evaluate_utterance(
    utterance: BenchmarkUtterance,
    prediction: SystemPrediction,
) -> UtteranceEvaluation:
    if utterance.utterance_id != prediction.utterance_id:
        raise ValueError("utterance and prediction IDs do not match")

    reading_pair_available = (
        utterance.reference_reading is not None and prediction.reading is not None
    )
    report = evaluate_japanese_asr(
        utterance.reference_text,
        prediction.text,
        reference_reading=(
            utterance.reference_reading if reading_pair_available else None
        ),
        hypothesis_reading=(prediction.reading if reading_pair_available else None),
    )

    learner: LearnerErrorPreservation | None = None
    if (
        utterance.target_reading is not None
        and utterance.observed_reading is not None
        and prediction.reading is not None
    ):
        learner = learner_error_preservation(
            utterance.target_reading,
            utterance.observed_reading,
            prediction.reading,
        )

    oracle_cer, oracle_cer_id = _best_error_candidate(
        prediction.candidates,
        reference_text=utterance.reference_text,
    )
    oracle_mora: ErrorRate | None = None
    oracle_mora_id: str | None = None
    if utterance.reference_reading is not None:
        oracle_mora, oracle_mora_id = _best_error_candidate(
            prediction.candidates,
            reference_reading=utterance.reference_reading,
        )

    return UtteranceEvaluation(
        utterance_id=utterance.utterance_id,
        speaker_id=utterance.speaker_id,
        split=utterance.split,
        system_id=prediction.system_id,
        status=prediction.status,
        reference_is_speech=utterance.is_speech,
        predicted_is_speech=prediction.status != "no_speech",
        cer=report.cer,
        kana_cer=report.kana_cer,
        mora_error_rate=report.mora_error_rate,
        learner_preservation=learner,
        oracle_cer=oracle_cer,
        oracle_mora_error_rate=oracle_mora,
        oracle_cer_candidate_id=oracle_cer_id,
        oracle_mora_candidate_id=oracle_mora_id,
        latency_ms=prediction.latency_ms,
    )


def evaluate_system(
    utterances: Iterable[BenchmarkUtterance],
    predictions: Iterable[SystemPrediction],
    *,
    require_complete: bool = True,
) -> tuple[UtteranceEvaluation, ...]:
    manifest = tuple(utterances)
    prediction_items = tuple(predictions)
    by_id: dict[str, SystemPrediction] = {}
    for prediction in prediction_items:
        if prediction.utterance_id in by_id:
            raise ValueError(f"duplicate prediction for {prediction.utterance_id!r}")
        by_id[prediction.utterance_id] = prediction

    manifest_ids = {item.utterance_id for item in manifest}
    extra_ids = set(by_id) - manifest_ids
    if extra_ids:
        raise ValueError(f"predictions contain unknown utterance IDs: {sorted(extra_ids)}")
    if require_complete:
        missing_ids = manifest_ids - set(by_id)
        if missing_ids:
            raise ValueError(f"missing predictions for utterance IDs: {sorted(missing_ids)}")

    evaluations = [
        evaluate_utterance(item, by_id[item.utterance_id])
        for item in manifest
        if item.utterance_id in by_id
    ]
    system_ids = {item.system_id for item in evaluations}
    if len(system_ids) > 1:
        raise ValueError("evaluate_system accepts predictions from one system at a time")
    return tuple(evaluations)


def _sum_error_rates(values: Iterable[ErrorRate]) -> ErrorRate:
    items = tuple(values)
    return ErrorRate(
        substitutions=sum(item.substitutions for item in items),
        deletions=sum(item.deletions for item in items),
        insertions=sum(item.insertions for item in items),
        reference_units=sum(item.reference_units for item in items),
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires values")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def aggregate_system_metrics(
    evaluations: Iterable[UtteranceEvaluation],
) -> AggregateSystemMetrics:
    items = tuple(evaluations)
    if not items:
        raise ValueError("at least one utterance evaluation is required")
    system_ids = {item.system_id for item in items}
    if len(system_ids) != 1:
        raise ValueError("all evaluations must belong to one system")

    kana = tuple(item.kana_cer for item in items if item.kana_cer is not None)
    mora = tuple(
        item.mora_error_rate for item in items if item.mora_error_rate is not None
    )
    oracle_cer = tuple(item.oracle_cer for item in items if item.oracle_cer is not None)
    oracle_mora = tuple(
        item.oracle_mora_error_rate
        for item in items
        if item.oracle_mora_error_rate is not None
    )
    learner = tuple(
        item.learner_preservation
        for item in items
        if item.learner_preservation is not None
    )
    preservation_scores = tuple(
        value.preservation_score
        for value in learner
        if value.preservation_score is not None
    )
    normalized_to_target = tuple(value.normalized_to_target for value in learner)
    latencies = tuple(
        item.latency_ms for item in items if item.latency_ms is not None
    )
    accepted_speech = tuple(
        item
        for item in items
        if item.reference_is_speech and item.status == "accept"
    )
    accepted_mora = tuple(
        item.mora_error_rate
        for item in accepted_speech
        if item.mora_error_rate is not None
    )

    reference_speech_count = sum(item.reference_is_speech for item in items)
    reference_no_speech_count = len(items) - reference_speech_count
    predicted_speech_count = sum(item.predicted_is_speech for item in items)
    true_speech_count = sum(
        item.reference_is_speech and item.predicted_is_speech for item in items
    )
    missed_speech_count = sum(
        item.reference_is_speech and not item.predicted_is_speech for item in items
    )
    true_no_speech_count = sum(
        not item.reference_is_speech and not item.predicted_is_speech for item in items
    )
    hallucinated_speech_count = sum(
        not item.reference_is_speech and item.predicted_is_speech for item in items
    )
    predicted_no_speech_count = len(items) - predicted_speech_count
    no_speech_precision = _safe_ratio(
        true_no_speech_count, predicted_no_speech_count
    )
    no_speech_recall = _safe_ratio(
        true_no_speech_count, reference_no_speech_count
    )

    return AggregateSystemMetrics(
        system_id=next(iter(system_ids)),
        utterance_count=len(items),
        cer=_sum_error_rates(item.cer for item in items),
        kana_cer=None if not kana else _sum_error_rates(kana),
        mora_error_rate=None if not mora else _sum_error_rates(mora),
        oracle_cer=None if not oracle_cer else _sum_error_rates(oracle_cer),
        oracle_mora_error_rate=(
            None if not oracle_mora else _sum_error_rates(oracle_mora)
        ),
        accepted_cer=(
            None
            if not accepted_speech
            else _sum_error_rates(item.cer for item in accepted_speech)
        ),
        accepted_mora_error_rate=(
            None if not accepted_mora else _sum_error_rates(accepted_mora)
        ),
        kana_evaluated=len(kana),
        mora_evaluated=len(mora),
        learner_evaluated=len(learner),
        oracle_cer_evaluated=len(oracle_cer),
        oracle_mora_evaluated=len(oracle_mora),
        accepted_speech_count=len(accepted_speech),
        mean_learner_preservation=(
            None if not preservation_scores else mean(preservation_scores)
        ),
        normalized_to_target_rate=(
            None
            if not normalized_to_target
            else sum(normalized_to_target) / len(normalized_to_target)
        ),
        accept_count=sum(item.status == "accept" for item in items),
        review_count=sum(item.status == "review" for item in items),
        no_speech_count=sum(item.status == "no_speech" for item in items),
        reference_speech_count=reference_speech_count,
        reference_no_speech_count=reference_no_speech_count,
        predicted_speech_count=predicted_speech_count,
        true_speech_count=true_speech_count,
        missed_speech_count=missed_speech_count,
        true_no_speech_count=true_no_speech_count,
        hallucinated_speech_count=hallucinated_speech_count,
        speech_accept_rate=_safe_ratio(len(accepted_speech), reference_speech_count),
        no_speech_precision=no_speech_precision,
        no_speech_recall=no_speech_recall,
        no_speech_f1=_f1(no_speech_precision, no_speech_recall),
        hallucination_on_silence_rate=_safe_ratio(
            hallucinated_speech_count, reference_no_speech_count
        ),
        missed_speech_rate=_safe_ratio(
            missed_speech_count, reference_speech_count
        ),
        latency_p50_ms=None if not latencies else _quantile(latencies, 0.50),
        latency_p95_ms=None if not latencies else _quantile(latencies, 0.95),
    )


def _evaluation_metric(
    evaluation: UtteranceEvaluation,
    metric: str,
) -> ErrorRate:
    if metric == "cer":
        return evaluation.cer
    if metric == "mora_error_rate":
        if evaluation.mora_error_rate is None:
            raise ValueError(
                f"mora_error_rate unavailable for {evaluation.utterance_id!r}"
            )
        return evaluation.mora_error_rate
    raise ValueError("metric must be 'cer' or 'mora_error_rate'")


def paired_speaker_bootstrap(
    baseline: Iterable[UtteranceEvaluation],
    candidate: Iterable[UtteranceEvaluation],
    *,
    metric: str = "cer",
    samples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> BootstrapComparison:
    """Paired cluster bootstrap over speakers, preserving within-speaker correlation."""

    if samples < 1:
        raise ValueError("samples must be >= 1")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")

    baseline_items = tuple(baseline)
    candidate_items = tuple(candidate)
    baseline_by_id = {item.utterance_id: item for item in baseline_items}
    candidate_by_id = {item.utterance_id: item for item in candidate_items}
    if len(baseline_by_id) != len(baseline_items):
        raise ValueError("baseline contains duplicate utterance IDs")
    if len(candidate_by_id) != len(candidate_items):
        raise ValueError("candidate contains duplicate utterance IDs")
    if not baseline_by_id or set(baseline_by_id) != set(candidate_by_id):
        raise ValueError("baseline and candidate must contain the same utterance IDs")

    by_speaker: dict[str, list[str]] = {}
    for utterance_id, baseline_item in baseline_by_id.items():
        candidate_item = candidate_by_id[utterance_id]
        if baseline_item.speaker_id != candidate_item.speaker_id:
            raise ValueError(f"speaker mismatch for {utterance_id!r}")
        by_speaker.setdefault(baseline_item.speaker_id, []).append(utterance_id)
    speakers = tuple(sorted(by_speaker))

    baseline_observed = _sum_error_rates(
        _evaluation_metric(item, metric) for item in baseline_by_id.values()
    ).rate
    candidate_observed = _sum_error_rates(
        _evaluation_metric(item, metric) for item in candidate_by_id.values()
    ).rate

    rng = Random(seed)
    deltas: list[float] = []
    for _ in range(samples):
        sampled_speakers = [rng.choice(speakers) for _ in speakers]
        baseline_rates: list[ErrorRate] = []
        candidate_rates: list[ErrorRate] = []
        for speaker_id in sampled_speakers:
            for utterance_id in by_speaker[speaker_id]:
                baseline_rates.append(
                    _evaluation_metric(baseline_by_id[utterance_id], metric)
                )
                candidate_rates.append(
                    _evaluation_metric(candidate_by_id[utterance_id], metric)
                )
        deltas.append(
            _sum_error_rates(candidate_rates).rate
            - _sum_error_rates(baseline_rates).rate
        )

    alpha = 1.0 - confidence_level
    return BootstrapComparison(
        metric=metric,
        baseline_rate=baseline_observed,
        candidate_rate=candidate_observed,
        delta=candidate_observed - baseline_observed,
        confidence_level=confidence_level,
        lower_bound=_quantile(deltas, alpha / 2.0),
        upper_bound=_quantile(deltas, 1.0 - alpha / 2.0),
        improvement_probability=sum(delta < 0.0 for delta in deltas) / len(deltas),
        samples=samples,
        cluster_count=len(speakers),
    )


def evaluate_accuracy_gate(
    baseline: AggregateSystemMetrics,
    candidate: AggregateSystemMetrics,
    *,
    config: AccuracyGateConfig = AccuracyGateConfig(),
    cer_bootstrap: BootstrapComparison | None = None,
    mora_bootstrap: BootstrapComparison | None = None,
) -> AccuracyGateResult:
    reasons: list[str] = []
    cer_delta = candidate.cer.rate - baseline.cer.rate
    if cer_delta > config.max_cer_regression:
        reasons.append("cer_regression")

    mora_delta: float | None = None
    if baseline.mora_error_rate is not None and candidate.mora_error_rate is not None:
        mora_delta = candidate.mora_error_rate.rate - baseline.mora_error_rate.rate
        if mora_delta > config.max_mora_regression:
            reasons.append("mora_error_rate_regression")

    preservation_delta: float | None = None
    if (
        baseline.mean_learner_preservation is not None
        and candidate.mean_learner_preservation is not None
    ):
        preservation_delta = (
            candidate.mean_learner_preservation
            - baseline.mean_learner_preservation
        )
        if preservation_delta < config.min_learner_preservation_delta:
            reasons.append("learner_error_preservation_regression")

    normalization_delta: float | None = None
    if (
        baseline.normalized_to_target_rate is not None
        and candidate.normalized_to_target_rate is not None
    ):
        normalization_delta = (
            candidate.normalized_to_target_rate
            - baseline.normalized_to_target_rate
        )
        if normalization_delta > config.max_normalized_to_target_increase:
            reasons.append("normalized_to_target_rate_regression")

    hallucination_delta: float | None = None
    if (
        baseline.hallucination_on_silence_rate is not None
        and candidate.hallucination_on_silence_rate is not None
    ):
        hallucination_delta = (
            candidate.hallucination_on_silence_rate
            - baseline.hallucination_on_silence_rate
        )
        if hallucination_delta > config.max_hallucination_on_silence_increase:
            reasons.append("hallucination_on_silence_regression")

    missed_speech_delta: float | None = None
    if baseline.missed_speech_rate is not None and candidate.missed_speech_rate is not None:
        missed_speech_delta = candidate.missed_speech_rate - baseline.missed_speech_rate
        if missed_speech_delta > config.max_missed_speech_rate_increase:
            reasons.append("missed_speech_rate_regression")

    speech_accept_delta: float | None = None
    if baseline.speech_accept_rate is not None and candidate.speech_accept_rate is not None:
        speech_accept_delta = candidate.speech_accept_rate - baseline.speech_accept_rate
        if speech_accept_delta < -config.max_speech_accept_rate_decrease:
            reasons.append("speech_accept_rate_regression")

    if config.require_cer_confidence_improvement:
        if cer_bootstrap is None:
            reasons.append("missing_cer_bootstrap")
        elif cer_bootstrap.upper_bound >= 0.0:
            reasons.append("cer_improvement_not_statistically_supported")
    if config.require_mora_confidence_improvement:
        if mora_bootstrap is None:
            reasons.append("missing_mora_bootstrap")
        elif mora_bootstrap.upper_bound >= 0.0:
            reasons.append("mora_improvement_not_statistically_supported")

    return AccuracyGateResult(
        passed=not reasons,
        reasons=tuple(reasons),
        deltas={
            "cer": cer_delta,
            "moraErrorRate": mora_delta,
            "learnerPreservation": preservation_delta,
            "normalizedToTargetRate": normalization_delta,
            "hallucinationOnSilenceRate": hallucination_delta,
            "missedSpeechRate": missed_speech_delta,
            "speechAcceptRate": speech_accept_delta,
        },
    )
