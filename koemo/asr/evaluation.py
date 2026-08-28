"""Japanese ASR evaluation metrics with learner-error preservation signals."""
from __future__ import annotations

from dataclasses import dataclass
import unicodedata
from typing import Sequence

from .mora import normalize_kana, split_mora


@dataclass(frozen=True, slots=True)
class ErrorRate:
    substitutions: int
    deletions: int
    insertions: int
    reference_units: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        if self.reference_units == 0:
            return 0.0 if self.errors == 0 else 1.0
        return self.errors / self.reference_units

    def to_dict(self) -> dict[str, int | float]:
        return {
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "errors": self.errors,
            "referenceUnits": self.reference_units,
            "rate": self.rate,
        }


@dataclass(frozen=True, slots=True)
class JapaneseASRReport:
    cer: ErrorRate
    kana_cer: ErrorRate | None
    mora_error_rate: ErrorRate | None


@dataclass(frozen=True, slots=True)
class LearnerErrorPreservation:
    target_to_observed: ErrorRate
    observed_to_hypothesis: ErrorRate
    target_to_hypothesis: ErrorRate
    preservation_score: float | None
    preservation_margin: float
    exact_observation: bool
    normalized_to_target: bool


def error_rate(reference: Sequence[str], hypothesis: Sequence[str]) -> ErrorRate:
    """Compute Levenshtein S/D/I counts with deterministic backtrace ties."""

    # Cell: (distance, substitutions, deletions, insertions).
    previous = [(index, 0, 0, index) for index in range(len(hypothesis) + 1)]
    for ref_index, ref_unit in enumerate(reference, start=1):
        current = [(ref_index, 0, ref_index, 0)]
        for hyp_index, hyp_unit in enumerate(hypothesis, start=1):
            if ref_unit == hyp_unit:
                current.append(previous[hyp_index - 1])
                continue
            substitution = previous[hyp_index - 1]
            deletion = previous[hyp_index]
            insertion = current[hyp_index - 1]
            candidates = (
                (
                    substitution[0] + 1,
                    substitution[1] + 1,
                    substitution[2],
                    substitution[3],
                ),
                (
                    deletion[0] + 1,
                    deletion[1],
                    deletion[2] + 1,
                    deletion[3],
                ),
                (
                    insertion[0] + 1,
                    insertion[1],
                    insertion[2],
                    insertion[3] + 1,
                ),
            )
            current.append(
                min(candidates, key=lambda value: (value[0], value[1], value[2], value[3]))
            )
        previous = current

    _, substitutions, deletions, insertions = previous[-1]
    return ErrorRate(
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_units=len(reference),
    )


def _normalize_characters(text: str, *, ignore_whitespace: bool) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text)
    return tuple(
        character
        for character in normalized
        if not (ignore_whitespace and character.isspace())
    )


def character_error_rate(
    reference: str,
    hypothesis: str,
    *,
    ignore_whitespace: bool = True,
) -> ErrorRate:
    return error_rate(
        _normalize_characters(reference, ignore_whitespace=ignore_whitespace),
        _normalize_characters(hypothesis, ignore_whitespace=ignore_whitespace),
    )


def kana_character_error_rate(reference_reading: str, hypothesis_reading: str) -> ErrorRate:
    reference = tuple(
        character
        for character in normalize_kana(reference_reading)
        if not character.isspace()
    )
    hypothesis = tuple(
        character
        for character in normalize_kana(hypothesis_reading)
        if not character.isspace()
    )
    return error_rate(reference, hypothesis)


def mora_error_rate(reference_reading: str, hypothesis_reading: str) -> ErrorRate:
    reference = split_mora(reference_reading)
    hypothesis = split_mora(hypothesis_reading)
    if reference_reading.strip() and not reference:
        raise ValueError("reference_reading contains no recognizable kana mora")
    if hypothesis_reading.strip() and not hypothesis:
        raise ValueError("hypothesis_reading contains no recognizable kana mora")
    return error_rate(reference, hypothesis)


def evaluate_japanese_asr(
    reference_text: str,
    hypothesis_text: str,
    *,
    reference_reading: str | None = None,
    hypothesis_reading: str | None = None,
) -> JapaneseASRReport:
    if (reference_reading is None) != (hypothesis_reading is None):
        raise ValueError("reference_reading and hypothesis_reading must be supplied together")
    kana = None
    mora = None
    if reference_reading is not None and hypothesis_reading is not None:
        kana = kana_character_error_rate(reference_reading, hypothesis_reading)
        mora = mora_error_rate(reference_reading, hypothesis_reading)
    return JapaneseASRReport(
        cer=character_error_rate(reference_text, hypothesis_text),
        kana_cer=kana,
        mora_error_rate=mora,
    )


def learner_error_preservation(
    target_reading: str,
    observed_reading: str,
    hypothesis_reading: str,
) -> LearnerErrorPreservation:
    """Measure whether ASR retained what the learner actually pronounced.

    ``preservation_score`` is 1 for an exact observed reading and falls to 0
    when ASR-to-observed distance reaches the learner's original deviation from
    the target. ``preservation_margin`` is positive when the ASR result is closer
    to the observation than to the intended target.
    """

    target = split_mora(target_reading)
    observed = split_mora(observed_reading)
    hypothesis = split_mora(hypothesis_reading)
    if not target or not observed or not hypothesis:
        raise ValueError("target, observed, and hypothesis must contain kana mora")

    target_observed = error_rate(target, observed)
    observed_hypothesis = error_rate(observed, hypothesis)
    target_hypothesis = error_rate(target, hypothesis)
    if target_observed.errors == 0:
        preservation_score = None
    else:
        preservation_score = max(
            0.0,
            1.0 - observed_hypothesis.errors / target_observed.errors,
        )

    normalizer = max(len(target), len(observed), len(hypothesis), 1)
    preservation_margin = (
        target_hypothesis.errors - observed_hypothesis.errors
    ) / normalizer
    return LearnerErrorPreservation(
        target_to_observed=target_observed,
        observed_to_hypothesis=observed_hypothesis,
        target_to_hypothesis=target_hypothesis,
        preservation_score=preservation_score,
        preservation_margin=preservation_margin,
        exact_observation=hypothesis == observed,
        normalized_to_target=target != observed and hypothesis == target,
    )
