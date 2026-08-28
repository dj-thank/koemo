"""Canonical data contracts for Koemo's mora-aware ASR pipeline.

The core rule is that acoustic observation and linguistic normalization are
stored separately.  No downstream normalizer may silently replace the text
selected from acoustic evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, Mapping


class UnitKind(str, Enum):
    """Semantic type of a canonical unit."""

    MORA = "mora"
    BOUNDARY = "boundary"
    NOISE = "noise"
    OTHER = "other"


class UnitSource(str, Enum):
    """Origin of a unit or alignment."""

    CHAR_CTC = "char_ctc"
    TEXT_READING = "text_reading"
    WHISPER = "whisper"
    QWEN_FORCED_ALIGNER = "qwen_forced_aligner"
    FUSED = "fused"


@dataclass(frozen=True, slots=True)
class TextSpan:
    """Half-open span in the source text: ``[start, end)``."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid text span: [{self.start}, {self.end})")


@dataclass(frozen=True, slots=True)
class TimeSpan:
    """Half-open time span in seconds: ``[start, end)``."""

    start: float
    end: float

    def __post_init__(self) -> None:
        if not isfinite(self.start) or not isfinite(self.end):
            raise ValueError("time span values must be finite")
        if self.start < 0.0 or self.end < self.start:
            raise ValueError(f"invalid time span: [{self.start}, {self.end})")

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class CTCCharUnit:
    """One character/token after (or before) CTC path collapse.

    ``is_blank`` lets the same type represent frame-level CTC paths.  When
    timestamps are unknown, both ``start`` and ``end`` must be ``None``.
    """

    symbol: str
    posterior: float = 1.0
    start: float | None = None
    end: float | None = None
    token_id: int | None = None
    frame_start: int | None = None
    frame_end: int | None = None
    is_blank: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.posterior <= 1.0:
            raise ValueError("posterior must be in [0, 1]")
        if (self.start is None) != (self.end is None):
            raise ValueError("start and end must both be set or both be None")
        if self.start is not None:
            TimeSpan(self.start, self.end if self.end is not None else self.start)
        if (self.frame_start is None) != (self.frame_end is None):
            raise ValueError("frame_start and frame_end must both be set or both be None")
        if self.frame_start is not None:
            if self.frame_start < 0 or self.frame_end is None or self.frame_end < self.frame_start:
                raise ValueError("invalid frame span")
        if not self.is_blank and not self.symbol:
            raise ValueError("non-blank CTC unit requires a symbol")

    @property
    def time_span(self) -> TimeSpan | None:
        if self.start is None or self.end is None:
            return None
        return TimeSpan(self.start, self.end)


@dataclass(frozen=True, slots=True)
class MoraUnit:
    """Canonical unit shared by CTC, Whisper, aligners and prosody heads.

    ``surface`` preserves the observed spelling. ``reading`` is the normalized
    katakana reading when known. ``mora`` is the atomic mora label; it is empty
    for boundaries and non-mora symbols.  ``phonemes`` describes phonetic
    realization and is intentionally optional because lexical mora and actual
    phones are not the same thing (e.g. devoicing and gemination).
    """

    unit_id: str
    surface: str
    reading: str
    mora: str
    kind: UnitKind
    source: UnitSource
    text_span: TextSpan
    time_span: TimeSpan | None = None
    posterior: float | None = None
    phonemes: tuple[str, ...] = ()
    alternatives: tuple[str, ...] = ()
    source_indices: tuple[int, ...] = ()
    accent_nucleus_probability: float | None = None
    accent_phrase_boundary_probability: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if not self.unit_id:
            raise ValueError("unit_id is required")
        if self.posterior is not None and not 0.0 <= self.posterior <= 1.0:
            raise ValueError("posterior must be in [0, 1]")
        for value, name in (
            (self.accent_nucleus_probability, "accent_nucleus_probability"),
            (
                self.accent_phrase_boundary_probability,
                "accent_phrase_boundary_probability",
            ),
        ):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.kind is UnitKind.MORA and not self.mora:
            raise ValueError("mora units require a non-empty mora label")
        if self.kind is not UnitKind.MORA and self.mora:
            raise ValueError("non-mora units must not carry a mora label")

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-ready representation for logs and APIs."""

        return {
            "unitId": self.unit_id,
            "surface": self.surface,
            "reading": self.reading,
            "mora": self.mora,
            "kind": self.kind.value,
            "source": self.source.value,
            "textSpan": {"start": self.text_span.start, "end": self.text_span.end},
            "audioSpan": (
                None
                if self.time_span is None
                else {"start": self.time_span.start, "end": self.time_span.end}
            ),
            "posterior": self.posterior,
            "phonemes": list(self.phonemes),
            "alternatives": list(self.alternatives),
            "sourceIndices": list(self.source_indices),
            "accent": {
                "nucleusProbability": self.accent_nucleus_probability,
                "phraseBoundaryProbability": self.accent_phrase_boundary_probability,
            },
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class HypothesisFeatures:
    """Higher-is-better evidence used for acoustic/lattice ranking.

    Raw model scores should be length-normalized before populating this object.
    Quality values are optional because stages are introduced incrementally.
    """

    whisper_logprob: float
    char_ctc_logprob: float | None = None
    mora_ctc_logprob: float | None = None
    alignment_quality: float | None = None
    prosody_quality: float | None = None
    no_speech_probability: float | None = None
    compression_ratio: float | None = None
    coverage: float | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.whisper_logprob, "whisper_logprob"),
            (self.char_ctc_logprob, "char_ctc_logprob"),
            (self.mora_ctc_logprob, "mora_ctc_logprob"),
        ):
            if value is not None and not isfinite(value):
                raise ValueError(f"{name} must be finite")
        for value, name in (
            (self.no_speech_probability, "no_speech_probability"),
            (self.alignment_quality, "alignment_quality"),
            (self.prosody_quality, "prosody_quality"),
            (self.coverage, "coverage"),
        ):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.compression_ratio is not None and self.compression_ratio < 0.0:
            raise ValueError("compression_ratio must be non-negative")


@dataclass(frozen=True, slots=True)
class TranscriptHypothesis:
    """A single ASR candidate and its provenance."""

    candidate_id: str
    text: str
    features: HypothesisFeatures
    token_ids: tuple[int, ...] = ()
    mora_units: tuple[MoraUnit, ...] = ()
    start: float | None = None
    end: float | None = None
    source: UnitSource = UnitSource.WHISPER

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id is required")
        if (self.start is None) != (self.end is None):
            raise ValueError("start and end must both be set or both be None")
        if self.start is not None and self.end is not None:
            TimeSpan(self.start, self.end)


@dataclass(frozen=True, slots=True)
class RankedHypothesis:
    """Candidate with calibrated ranking information."""

    hypothesis: TranscriptHypothesis
    acoustic_score: float
    rank: int
    component_scores: Mapping[str, float] = field(default_factory=dict, compare=False, hash=False)
    llm_rank: int | None = None
    llm_confidence: float | None = None
    llm_tiebreak_score: float = 0.0

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("rank is 1-based")
        if self.llm_rank is not None and self.llm_rank < 1:
            raise ValueError("llm_rank is 1-based")
        if self.llm_confidence is not None and not 0.0 <= self.llm_confidence <= 1.0:
            raise ValueError("llm_confidence must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class TranscriptState:
    """Pipeline state that keeps observation and normalization separate."""

    observed_transcript: str
    observed_candidate_id: str
    hypotheses: tuple[RankedHypothesis, ...]
    mora_units: tuple[MoraUnit, ...]
    normalized_transcript: str | None = None
    llm_preferred_candidate_id: str | None = None
    normalization_method: str | None = None

    def __post_init__(self) -> None:
        ids = {item.hypothesis.candidate_id for item in self.hypotheses}
        if self.observed_candidate_id not in ids:
            raise ValueError("observed_candidate_id is not present in hypotheses")
        if (
            self.llm_preferred_candidate_id is not None
            and self.llm_preferred_candidate_id not in ids
        ):
            raise ValueError("llm_preferred_candidate_id is not present in hypotheses")

    def to_dict(self) -> dict[str, Any]:
        return {
            "observedTranscript": self.observed_transcript,
            "observedCandidateId": self.observed_candidate_id,
            "normalizedTranscript": self.normalized_transcript,
            "normalizationMethod": self.normalization_method,
            "llmPreferredCandidateId": self.llm_preferred_candidate_id,
            "moraUnits": [unit.to_dict() for unit in self.mora_units],
            "hypotheses": [
                {
                    "candidateId": item.hypothesis.candidate_id,
                    "text": item.hypothesis.text,
                    "rank": item.rank,
                    "acousticScore": item.acoustic_score,
                    "componentScores": dict(item.component_scores),
                    "llmRank": item.llm_rank,
                    "llmConfidence": item.llm_confidence,
                    "llmTiebreakScore": item.llm_tiebreak_score,
                }
                for item in self.hypotheses
            ],
        }
