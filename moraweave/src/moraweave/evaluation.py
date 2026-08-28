from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Sequence

from .calibration import (
    area_under_risk_coverage,
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
)

NUMBER_PATTERN = re.compile(r"(?:\d[\d,.:/-]*|[〇一二三四五六七八九十百千万億兆]+)")
PUNCTUATION_PATTERN = re.compile(r"[、。！？!?：:；;]")
CRITICAL_ENTITY_PATTERN = re.compile(
    r"(?:\d[\d,.:/-]*|[A-Za-z][A-Za-z0-9+._/#-]*|[\u30A0-\u30FFー]{2,}|[\u3400-\u9FFF]{2,})"
)
FILLERS = ("えー", "ええと", "えっと", "あの", "その", "まあ", "うーん", "んー")


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, reference_item in enumerate(reference, 1):
        current = [row]
        for column, hypothesis_item in enumerate(hypothesis, 1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (reference_item != hypothesis_item),
                )
            )
        previous = current
    return previous[-1]


def error_rate(reference: Sequence[str], hypothesis: Sequence[str]) -> float | None:
    if not reference:
        return 0.0 if not hypothesis else None
    return edit_distance(reference, hypothesis) / len(reference)


def normalize_characters(text: str) -> list[str]:
    value = unicodedata.normalize("NFKC", str(text or ""))
    return [character for character in value if not character.isspace()]


def cer(reference: str, hypothesis: str) -> float | None:
    return error_rate(normalize_characters(reference), normalize_characters(hypothesis))


def kana_cer(reference_reading: str | None, hypothesis_reading: str | None) -> float | None:
    if reference_reading is None or hypothesis_reading is None:
        return None
    return cer(reference_reading, hypothesis_reading)


def mora_error_rate(
    reference_mora: Sequence[str] | None,
    hypothesis_mora: Sequence[str] | None,
) -> float | None:
    if reference_mora is None or hypothesis_mora is None:
        return None
    return error_rate(reference_mora, hypothesis_mora)


def _pattern_error_rate(pattern: re.Pattern[str], reference: str, hypothesis: str) -> float | None:
    expected = pattern.findall(unicodedata.normalize("NFKC", reference))
    observed = pattern.findall(unicodedata.normalize("NFKC", hypothesis))
    if not expected:
        return 0.0 if not observed else None
    return error_rate(expected, observed)


def number_error_rate(reference: str, hypothesis: str) -> float | None:
    return _pattern_error_rate(NUMBER_PATTERN, reference, hypothesis)


def punctuation_error_rate(reference: str, hypothesis: str) -> float | None:
    return _pattern_error_rate(PUNCTUATION_PATTERN, reference, hypothesis)


def critical_entity_sequence(text: str) -> list[str]:
    return CRITICAL_ENTITY_PATTERN.findall(unicodedata.normalize("NFKC", str(text or "")))


def critical_entity_error_rate(reference: str, hypothesis: str) -> float | None:
    expected = critical_entity_sequence(reference)
    observed = critical_entity_sequence(hypothesis)
    if not expected:
        return 0.0 if not observed else None
    return error_rate(expected, observed)


def filler_sequence(text: str, fillers: Iterable[str] = FILLERS) -> list[str]:
    found: list[tuple[int, str]] = []
    for filler in fillers:
        start = 0
        while True:
            index = text.find(filler, start)
            if index < 0:
                break
            found.append((index, filler))
            start = index + len(filler)
    return [value for _, value in sorted(found)]


def disfluency_preservation_rate(reference: str, hypothesis: str) -> float | None:
    expected = filler_sequence(reference)
    observed = filler_sequence(hypothesis)
    if not expected:
        return 1.0 if not observed else None
    return max(0.0, 1.0 - edit_distance(expected, observed) / len(expected))


def unsupported_correction_rate(
    observed_text: str,
    normalized_text: str,
    supported_spans: Sequence[tuple[int, int]] = (),
) -> float:
    """Return the fraction of edits outside explicitly supported observed spans."""

    observed = normalize_characters(observed_text)
    normalized = normalize_characters(normalized_text)
    if observed == normalized:
        return 0.0
    allowed: set[int] = set()
    for start, end in supported_spans:
        allowed.update(range(max(0, start), max(start, end)))

    rows, columns = len(observed) + 1, len(normalized) + 1
    dynamic: list[list[tuple[int, int]]] = [[(0, 0)] * columns for _ in range(rows)]
    for row in range(1, rows):
        dynamic[row][0] = (row, 0 if row - 1 in allowed else 1)
    for column in range(1, columns):
        dynamic[0][column] = (column, 1)
    for row in range(1, rows):
        for column in range(1, columns):
            changed = observed[row - 1] != normalized[column - 1]
            substitution_unsupported = int(changed and row - 1 not in allowed)
            options = [
                (
                    dynamic[row - 1][column][0] + 1,
                    dynamic[row - 1][column][1] + int(row - 1 not in allowed),
                ),
                (dynamic[row][column - 1][0] + 1, dynamic[row][column - 1][1] + 1),
                (
                    dynamic[row - 1][column - 1][0] + int(changed),
                    dynamic[row - 1][column - 1][1] + substitution_unsupported,
                ),
            ]
            dynamic[row][column] = min(options, key=lambda item: (item[0], item[1]))
    edits, unsupported = dynamic[-1][-1]
    return 0.0 if edits == 0 else unsupported / edits


def oracle_cer(reference: str, hypotheses: Sequence[str]) -> float | None:
    values = [cer(reference, hypothesis) for hypothesis in hypotheses]
    finite = [value for value in values if value is not None]
    return min(finite) if finite else None


def top_k_exact_match(reference: str, hypotheses: Sequence[str]) -> bool:
    expected = normalize_characters(reference)
    return any(normalize_characters(hypothesis) == expected for hypothesis in hypotheses)


@dataclass(frozen=True, slots=True)
class ConfidenceEvaluation:
    expected_calibration_error: float
    brier: float
    negative_log_likelihood: float
    aurc: float


def evaluate_confidence(
    confidences: Sequence[float],
    correct: Sequence[int | bool],
    *,
    bins: int = 15,
) -> ConfidenceEvaluation:
    return ConfidenceEvaluation(
        expected_calibration_error=expected_calibration_error(confidences, correct, bins=bins),
        brier=brier_score(confidences, correct),
        negative_log_likelihood=negative_log_likelihood(confidences, correct),
        aurc=area_under_risk_coverage(confidences, correct),
    )


def relisten_efficiency(
    before_error: float | None,
    after_error: float | None,
    *,
    cost_ms: int,
) -> float | None:
    if before_error is None or after_error is None or cost_ms <= 0:
        return None
    return (before_error - after_error) / cost_ms


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    cer: float | None
    kana_cer: float | None
    mler: float | None
    number_error_rate: float | None
    disfluency_preservation: float | None
    unsupported_correction: float
    punctuation_error_rate: float | None = None
    critical_entity_error_rate: float | None = None


def evaluate_transcript(
    *,
    reference: str,
    observed: str,
    normalized: str,
    reference_reading: str | None = None,
    observed_reading: str | None = None,
    reference_mora: Sequence[str] | None = None,
    observed_mora: Sequence[str] | None = None,
    supported_normalization_spans: Sequence[tuple[int, int]] = (),
) -> EvaluationResult:
    return EvaluationResult(
        cer=cer(reference, observed),
        kana_cer=kana_cer(reference_reading, observed_reading),
        mler=mora_error_rate(reference_mora, observed_mora),
        number_error_rate=number_error_rate(reference, observed),
        disfluency_preservation=disfluency_preservation_rate(reference, observed),
        unsupported_correction=unsupported_correction_rate(
            observed, normalized, supported_normalization_spans
        ),
        punctuation_error_rate=punctuation_error_rate(reference, observed),
        critical_entity_error_rate=critical_entity_error_rate(reference, observed),
    )
