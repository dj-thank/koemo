from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

EvidenceName = Literal["acoustic", "mora", "lexical", "preservation"]


def _canonical_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _canonical_value(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, set):
        return sorted(_canonical_value(item) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("canonical evidence cannot contain NaN or infinity")
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _finite_optional(value: float | None, *, name: str) -> None:
    if value is not None and not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class MoraUnit:
    kana: str
    start_ms: float | None = None
    end_ms: float | None = None
    confidence: float | None = None
    phones: tuple[str, ...] = ()
    kind: Literal["regular", "moraic-nasal", "geminate", "long-vowel"] = "regular"

    def __post_init__(self) -> None:
        if not self.kana:
            raise ValueError("mora kana must not be empty")
        for name, value in (
            ("start_ms", self.start_ms),
            ("end_ms", self.end_ms),
            ("confidence", self.confidence),
        ):
            _finite_optional(value, name=name)
        if self.start_ms is not None and self.start_ms < 0:
            raise ValueError("mora start_ms must be non-negative")
        if self.end_ms is not None and self.end_ms < 0:
            raise ValueError("mora end_ms must be non-negative")
        if self.start_ms is not None and self.end_ms is not None and self.end_ms < self.start_ms:
            raise ValueError("mora end_ms must be >= start_ms")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("mora confidence must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    candidate_id: str
    text: str
    token_ids: tuple[int, ...] = ()
    acoustic: float | None = None
    mora: float | None = None
    lexical: float | None = None
    preservation: float | None = None
    teacher: float | None = None
    mora_units: tuple[MoraUnit, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    rank: int | None = None
    hypothesis_count: int | None = None
    sequence_score: float | None = None
    avg_logprob: float | None = None
    beam_confidence: float | None = None
    source: str | None = None
    reading: str | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty")
        if not self.text:
            raise ValueError("candidate text must not be empty")
        if self.rank is not None and self.rank < 1:
            raise ValueError("candidate rank is one-based")
        if self.hypothesis_count is not None and self.hypothesis_count < 1:
            raise ValueError("hypothesis_count must be positive")
        if self.rank is not None and self.hypothesis_count is not None and self.hypothesis_count < self.rank:
            raise ValueError("hypothesis_count must be >= rank")
        for name, value in (
            ("acoustic", self.acoustic),
            ("mora", self.mora),
            ("lexical", self.lexical),
            ("preservation", self.preservation),
            ("teacher", self.teacher),
            ("sequence_score", self.sequence_score),
            ("avg_logprob", self.avg_logprob),
            ("beam_confidence", self.beam_confidence),
        ):
            _finite_optional(value, name=name)
        if self.beam_confidence is not None and not 0 <= self.beam_confidence <= 1:
            raise ValueError("beam_confidence must be in [0, 1]")

    def score(self, name: EvidenceName) -> float | None:
        return getattr(self, name)

    @property
    def evidence_source(self) -> str:
        return self.source or str(self.metadata.get("adapter") or "unknown")

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "CandidateEvidence":
        allowed = {item.name for item in dataclasses.fields(cls)}
        values = {key: row[key] for key in allowed if key in row}
        values["token_ids"] = tuple(int(value) for value in values.get("token_ids", ()))
        values["mora_units"] = tuple(
            value if isinstance(value, MoraUnit) else MoraUnit(**value)
            for value in values.get("mora_units", ())
        )
        values["metadata"] = dict(values.get("metadata", {}))
        return cls(**values)


@dataclass(frozen=True, slots=True)
class GateDecision:
    weights: dict[EvidenceName, float]
    disagreement: float
    entropy: float
    needs_relisten: bool
    reasons: tuple[str, ...] = ()
    posterior: dict[str, float] = field(default_factory=dict)
    evidence_coverage: float = 0.0
    selective_risk: float = 1.0
    abstain: bool = False
    calibration_digest: str | None = None
    uncertainty: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    candidate: CandidateEvidence
    final_score: float
    normalized_scores: dict[str, float]
    gate: GateDecision
    grammar_honeytrap_penalty: float = 0.0
    posterior: float = 0.0
    missing_evidence_penalty: float = 0.0
    source_diversity_bonus: float = 0.0


@dataclass(frozen=True, slots=True)
class ObservedTranscript:
    text: str
    selected_candidate_id: str
    candidates: tuple[CandidateEvidence, ...]
    ranked: tuple[RankedCandidate, ...]
    uncertainty_spans: tuple[dict[str, Any], ...]
    source_audio_sha256: str | None
    evidence_sha256: str
    decision: Literal["accepted", "provisional"] = "accepted"
    selected_posterior: float = 0.0

    @classmethod
    def create(
        cls,
        *,
        selected: RankedCandidate,
        ranked: list[RankedCandidate],
        uncertainty_spans: list[dict[str, Any]],
        source_audio_sha256: str | None = None,
    ) -> "ObservedTranscript":
        candidates = tuple(item.candidate for item in ranked)
        decision: Literal["accepted", "provisional"] = "provisional" if selected.gate.abstain else "accepted"
        payload = {
            "text": selected.candidate.text,
            "selectedCandidateId": selected.candidate.candidate_id,
            "candidates": candidates,
            "ranked": tuple(ranked),
            "uncertaintySpans": tuple(uncertainty_spans),
            "sourceAudioSha256": source_audio_sha256,
            "decision": decision,
            "selectedPosterior": selected.posterior,
        }
        return cls(
            text=selected.candidate.text,
            selected_candidate_id=selected.candidate.candidate_id,
            candidates=candidates,
            ranked=tuple(ranked),
            uncertainty_spans=tuple(uncertainty_spans),
            source_audio_sha256=source_audio_sha256,
            evidence_sha256=sha256_json(payload),
            decision=decision,
            selected_posterior=selected.posterior,
        )

    def verify(self) -> None:
        payload = {
            "text": self.text,
            "selectedCandidateId": self.selected_candidate_id,
            "candidates": self.candidates,
            "ranked": self.ranked,
            "uncertaintySpans": self.uncertainty_spans,
            "sourceAudioSha256": self.source_audio_sha256,
            "decision": self.decision,
            "selectedPosterior": self.selected_posterior,
        }
        if sha256_json(payload) != self.evidence_sha256:
            raise ValueError("observed transcript evidence was modified")
        selected = next((item for item in self.candidates if item.candidate_id == self.selected_candidate_id), None)
        if selected is None or selected.text != self.text:
            raise ValueError("observed text is detached from the selected acoustic candidate")


@dataclass(frozen=True, slots=True)
class NormalizedTranscript:
    text: str
    observed_evidence_sha256: str
    mode: Literal["deterministic", "rank-only", "guarded-rewrite"]
    selected_candidate_id: str | None = None
    rejected_edits: tuple[str, ...] = ()

    @classmethod
    def attach(
        cls,
        observed: ObservedTranscript,
        *,
        text: str,
        mode: Literal["deterministic", "rank-only", "guarded-rewrite"],
        selected_candidate_id: str | None = None,
        rejected_edits: tuple[str, ...] = (),
    ) -> "NormalizedTranscript":
        observed.verify()
        if selected_candidate_id is not None and selected_candidate_id not in {
            candidate.candidate_id for candidate in observed.candidates
        }:
            raise ValueError("normalization selected a candidate outside the observed lattice")
        return cls(
            text=text,
            observed_evidence_sha256=observed.evidence_sha256,
            mode=mode,
            selected_candidate_id=selected_candidate_id,
            rejected_edits=rejected_edits,
        )
