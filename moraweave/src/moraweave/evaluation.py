from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Sequence

NUMBER_PATTERN = re.compile(r"(?:\d[\d,.:/-]*|[〇一二三四五六七八九十百千万億兆]+)")
FILLERS = ("えー", "ええと", "えっと", "あの", "その", "まあ", "うーん", "んー")


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, ref_item in enumerate(reference, 1):
        current = [row]
        for column, hyp_item in enumerate(hypothesis, 1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (ref_item != hyp_item),
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
    return [char for char in value if not char.isspace()]


def cer(reference: str, hypothesis: str) -> float | None:
    return error_rate(normalize_characters(reference), normalize_characters(hypothesis))


def kana_cer(reference_reading: str | None, hypothesis_reading: str | None) -> float | None:
    if reference_reading is None or hypothesis_reading is None:
        return None
    return cer(reference_reading, hypothesis_reading)


def mora_error_rate(reference_mora: Sequence[str] | None, hypothesis_mora: Sequence[str] | None) -> float | None:
    if reference_mora is None or hypothesis_mora is None:
        return None
    return error_rate(reference_mora, hypothesis_mora)


def number_error_rate(reference: str, hypothesis: str) -> float | None:
    ref_numbers = NUMBER_PATTERN.findall(unicodedata.normalize("NFKC", reference))
    hyp_numbers = NUMBER_PATTERN.findall(unicodedata.normalize("NFKC", hypothesis))
    if not ref_numbers:
        return 0.0 if not hyp_numbers else None
    return error_rate(ref_numbers, hyp_numbers)


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
    distance = edit_distance(expected, observed)
    return max(0.0, 1.0 - distance / len(expected))


def unsupported_correction_rate(
    observed_text: str,
    normalized_text: str,
    supported_spans: Sequence[tuple[int, int]] = (),
) -> float:
    """Estimate edits outside explicitly supported observed-character spans."""

    observed = normalize_characters(observed_text)
    normalized = normalize_characters(normalized_text)
    if observed == normalized:
        return 0.0
    allowed = set()
    for start, end in supported_spans:
        allowed.update(range(max(0, start), max(start, end)))

    # A lightweight alignment that counts changed observed positions not covered by evidence.
    rows = len(observed) + 1
    cols = len(normalized) + 1
    dp: list[list[tuple[int, int]]] = [[(0, 0)] * cols for _ in range(rows)]
    for i in range(1, rows):
        dp[i][0] = (i, 0 if i - 1 in allowed else 1)
    for j in range(1, cols):
        dp[0][j] = (j, 1)
    for i in range(1, rows):
        for j in range(1, cols):
            match_cost = 0 if observed[i - 1] == normalized[j - 1] else 1
            unsupported = 0 if match_cost == 0 or i - 1 in allowed else 1
            choices = [
                (dp[i - 1][j][0] + 1, dp[i - 1][j][1] + (0 if i - 1 in allowed else 1)),
                (dp[i][j - 1][0] + 1, dp[i][j - 1][1] + 1),
                (dp[i - 1][j - 1][0] + match_cost, dp[i - 1][j - 1][1] + unsupported),
            ]
            dp[i][j] = min(choices, key=lambda item: (item[0], item[1]))
    edits, unsupported_edits = dp[-1][-1]
    return 0.0 if edits == 0 else unsupported_edits / edits


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    cer: float | None
    kana_cer: float | None
    mler: float | None
    number_error_rate: float | None
    disfluency_preservation: float | None
    unsupported_correction: float


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
    )
