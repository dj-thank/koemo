from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

from .contracts import CandidateEvidence, NormalizedTranscript, ObservedTranscript
from .gates import GateConfig, evidence_summary, gate_candidates
from .selective import RelistenRequest, merge_relisten_candidates, plan_relisten


@dataclass(frozen=True, slots=True)
class PipelineResult:
    observed: ObservedTranscript
    normalized: NormalizedTranscript | None
    relisten_requests: tuple[RelistenRequest, ...]
    diagnostics: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class MoraWeavePipeline:
    def __init__(self, gate_config: GateConfig | None = None) -> None:
        self.gate_config = gate_config or GateConfig.default()

    def observe(
        self,
        candidates: list[CandidateEvidence],
        *,
        source_audio_sha256: str | None = None,
        segment_start_ms: int = 0,
        segment_end_ms: int = 30_000,
        token_spans: list[dict[str, object]] | None = None,
        relisten: Callable[[RelistenRequest], list[CandidateEvidence]] | None = None,
    ) -> tuple[ObservedTranscript, tuple[RelistenRequest, ...], dict[str, object]]:
        ranked = gate_candidates(candidates, self.gate_config)
        requests = plan_relisten(
            ranked,
            segment_start_ms=segment_start_ms,
            segment_end_ms=segment_end_ms,
            token_spans=token_spans,
        )

        relisten_count = 0
        if relisten is not None and requests:
            additional: list[CandidateEvidence] = []
            for request in requests:
                additional.extend(relisten(request))
            if additional:
                candidates = merge_relisten_candidates(candidates, additional)
                ranked = gate_candidates(candidates, self.gate_config)
                relisten_count = len(additional)

        uncertainty_spans = [
            {
                "startMs": request.span.start_ms,
                "endMs": request.span.end_ms,
                "reasons": list(request.span.reasons),
                "priority": request.span.priority,
            }
            for request in requests
        ]
        observed = ObservedTranscript.create(
            selected=ranked[0],
            ranked=ranked,
            uncertainty_spans=uncertainty_spans,
            source_audio_sha256=source_audio_sha256,
        )
        observed.verify()
        diagnostics = {
            **evidence_summary(ranked),
            "candidateCount": len(ranked),
            "relistenRequestCount": len(requests),
            "additionalCandidateCount": relisten_count,
        }
        return observed, tuple(requests), diagnostics

    def normalize_rank_only(
        self,
        observed: ObservedTranscript,
        ordered_candidate_ids: list[str],
    ) -> NormalizedTranscript:
        observed.verify()
        expected = {candidate.candidate_id for candidate in observed.candidates}
        if len(ordered_candidate_ids) != len(expected) or set(ordered_candidate_ids) != expected:
            raise ValueError("rank-only normalizer must return every candidate ID exactly once")
        selected_id = ordered_candidate_ids[0]
        selected = next(
            candidate for candidate in observed.candidates if candidate.candidate_id == selected_id
        )
        return NormalizedTranscript.attach(
            observed,
            text=selected.text,
            mode="rank-only",
            selected_candidate_id=selected_id,
        )

    def run(
        self,
        candidates: list[CandidateEvidence],
        *,
        source_audio_sha256: str | None = None,
        segment_start_ms: int = 0,
        segment_end_ms: int = 30_000,
        token_spans: list[dict[str, object]] | None = None,
        relisten: Callable[[RelistenRequest], list[CandidateEvidence]] | None = None,
        local_rank: list[str] | None = None,
    ) -> PipelineResult:
        observed, requests, diagnostics = self.observe(
            candidates,
            source_audio_sha256=source_audio_sha256,
            segment_start_ms=segment_start_ms,
            segment_end_ms=segment_end_ms,
            token_spans=token_spans,
            relisten=relisten,
        )
        normalized = (
            self.normalize_rank_only(observed, local_rank) if local_rank is not None else None
        )
        return PipelineResult(
            observed=observed,
            normalized=normalized,
            relisten_requests=requests,
            diagnostics=diagnostics,
        )
