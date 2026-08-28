from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

EvidenceName = Literal["acoustic", "mora", "lexical", "preservation"]


def canonical_json(value: Any) -> str:
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class MoraUnit:
    kana: str
    start_ms: float | None = None
    end_ms: float | None = None
    confidence: float | None = None
    phones: tuple[str, ...] = ()
    kind: Literal["regular", "moraic-nasal", "geminate", "long-vowel"] = "regular"


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

    def score(self, name: EvidenceName) -> float | None:
        return getattr(self, name)


@dataclass(frozen=True, slots=True)
class GateDecision:
    weights: dict[EvidenceName, float]
    disagreement: float
    entropy: float
    needs_relisten: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    candidate: CandidateEvidence
    final_score: float
    normalized_scores: dict[str, float]
    gate: GateDecision
    grammar_honeytrap_penalty: float = 0.0


@dataclass(frozen=True, slots=True)
class ObservedTranscript:
    text: str
    selected_candidate_id: str
    candidates: tuple[CandidateEvidence, ...]
    ranked: tuple[RankedCandidate, ...]
    uncertainty_spans: tuple[dict[str, Any], ...]
    source_audio_sha256: str | None
    evidence_sha256: str

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
        payload = {
            "text": selected.candidate.text,
            "selectedCandidateId": selected.candidate.candidate_id,
            "candidates": candidates,
            "ranked": tuple(ranked),
            "uncertaintySpans": tuple(uncertainty_spans),
            "sourceAudioSha256": source_audio_sha256,
        }
        return cls(
            text=selected.candidate.text,
            selected_candidate_id=selected.candidate.candidate_id,
            candidates=candidates,
            ranked=tuple(ranked),
            uncertainty_spans=tuple(uncertainty_spans),
            source_audio_sha256=source_audio_sha256,
            evidence_sha256=sha256_json(payload),
        )

    def verify(self) -> None:
        payload = {
            "text": self.text,
            "selectedCandidateId": self.selected_candidate_id,
            "candidates": self.candidates,
            "ranked": self.ranked,
            "uncertaintySpans": self.uncertainty_spans,
            "sourceAudioSha256": self.source_audio_sha256,
        }
        if sha256_json(payload) != self.evidence_sha256:
            raise ValueError("observed transcript evidence was modified")
        selected = next(
            (item for item in self.candidates if item.candidate_id == self.selected_candidate_id),
            None,
        )
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
