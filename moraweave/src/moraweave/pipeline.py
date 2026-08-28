from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

from .acquisition import (
    EvidenceAction,
    EvidenceAcquisitionPlan,
    EvidenceBudget,
    plan_evidence_acquisition,
)
from .contracts import CandidateEvidence, NormalizedTranscript, ObservedTranscript
from .gates import GateConfig, evidence_summary, gate_candidates
from .selective import RelistenRequest, merge_relisten_candidates, plan_relisten
from .shadow_lattice import (
    DualEvidenceLattice,
    build_dual_evidence_lattice,
    lattice_token_spans,
)


@dataclass(frozen=True, slots=True)
class PipelineResult:
    observed: ObservedTranscript
    normalized: NormalizedTranscript | None
    relisten_requests: tuple[RelistenRequest, ...]
    evidence_actions: tuple[EvidenceAction, ...]
    diagnostics: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class MoraWeavePipeline:
    def __init__(
        self,
        gate_config: GateConfig | None = None,
        *,
        evidence_budget: EvidenceBudget | None = None,
    ) -> None:
        self.gate_config = gate_config or GateConfig.default()
        self.evidence_budget = evidence_budget or EvidenceBudget()

    @staticmethod
    def _lattice(
        candidates: list[CandidateEvidence],
        ranked,
        *,
        timeline: list[dict[str, object]] | None,
    ) -> DualEvidenceLattice:
        return build_dual_evidence_lattice(
            candidates,
            posterior=ranked[0].gate.posterior,
            pivot_candidate_id=ranked[0].candidate.candidate_id,
            timeline=timeline,
        )

    def observe_with_plan(
        self,
        candidates: list[CandidateEvidence],
        *,
        source_audio_sha256: str | None = None,
        segment_start_ms: int = 0,
        segment_end_ms: int = 30_000,
        token_spans: list[dict[str, object]] | None = None,
        lattice_timeline: list[dict[str, object]] | None = None,
        relisten: Callable[[RelistenRequest], list[CandidateEvidence]] | None = None,
        evidence_acquire: Callable[[EvidenceAction], list[CandidateEvidence]] | None = None,
    ) -> tuple[
        ObservedTranscript,
        tuple[RelistenRequest, ...],
        tuple[EvidenceAction, ...],
        dict[str, object],
    ]:
        ranked = gate_candidates(candidates, self.gate_config)
        lattice = self._lattice(candidates, ranked, timeline=lattice_timeline)
        localized_spans = lattice_token_spans(lattice)
        planning_spans = [*(token_spans or []), *localized_spans]
        requests = plan_relisten(
            ranked,
            segment_start_ms=segment_start_ms,
            segment_end_ms=segment_end_ms,
            token_spans=planning_spans or None,
        )
        acquisition_plan: EvidenceAcquisitionPlan = plan_evidence_acquisition(
            ranked,
            lattice,
            budget=self.evidence_budget,
        )

        additional: list[CandidateEvidence] = []
        executed_action_ids: list[str] = []
        if evidence_acquire is not None:
            for action in acquisition_plan.selected:
                if not action.affects_observed_decision:
                    continue
                rows = evidence_acquire(action)
                if rows:
                    additional.extend(rows)
                    executed_action_ids.append(action.action_id)

        if relisten is not None and requests:
            for request in requests:
                rows = relisten(request)
                if rows:
                    additional.extend(rows)

        if additional:
            candidates = merge_relisten_candidates(candidates, additional)
            ranked = gate_candidates(candidates, self.gate_config)
            lattice = self._lattice(candidates, ranked, timeline=lattice_timeline)

        uncertainty_spans = [
            {
                "startMs": request.span.start_ms,
                "endMs": request.span.end_ms,
                "reasons": list(request.span.reasons),
                "priority": request.span.priority,
                "source": "legacy-relisten-planner",
            }
            for request in requests
        ]
        for action in acquisition_plan.selected:
            if action.start_ms is None or action.end_ms is None:
                continue
            uncertainty_spans.append(
                {
                    "startMs": action.start_ms,
                    "endMs": action.end_ms,
                    "reasons": list(action.reasons),
                    "priority": action.expected_information_gain,
                    "source": action.kind,
                    "actionId": action.action_id,
                }
            )

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
            "additionalCandidateCount": len(additional),
            "observationDecision": observed.decision,
            "latticeAlignmentLevel": lattice.alignment_level,
            "consensusSpanCount": len(lattice.locked_consensus),
            "contradictionIslandCount": len(lattice.contradiction_islands),
            "evidenceBudgetMs": acquisition_plan.budget_ms,
            "evidenceBudgetUsedMs": acquisition_plan.used_ms,
            "plannedEvidenceActionCount": len(acquisition_plan.selected),
            "plannedInformationGain": acquisition_plan.expected_information_gain,
            "evidenceStoppingReason": acquisition_plan.stopping_reason,
            "executedEvidenceActionIds": executed_action_ids,
        }
        return observed, tuple(requests), acquisition_plan.selected, diagnostics

    def observe(
        self,
        candidates: list[CandidateEvidence],
        *,
        source_audio_sha256: str | None = None,
        segment_start_ms: int = 0,
        segment_end_ms: int = 30_000,
        token_spans: list[dict[str, object]] | None = None,
        lattice_timeline: list[dict[str, object]] | None = None,
        relisten: Callable[[RelistenRequest], list[CandidateEvidence]] | None = None,
        evidence_acquire: Callable[[EvidenceAction], list[CandidateEvidence]] | None = None,
    ) -> tuple[ObservedTranscript, tuple[RelistenRequest, ...], dict[str, object]]:
        observed, requests, _actions, diagnostics = self.observe_with_plan(
            candidates,
            source_audio_sha256=source_audio_sha256,
            segment_start_ms=segment_start_ms,
            segment_end_ms=segment_end_ms,
            token_spans=token_spans,
            lattice_timeline=lattice_timeline,
            relisten=relisten,
            evidence_acquire=evidence_acquire,
        )
        return observed, requests, diagnostics

    def normalize_rank_only(
        self,
        observed: ObservedTranscript,
        ordered_candidate_ids: list[str],
    ) -> NormalizedTranscript:
        observed.verify()
        expected = {candidate.candidate_id for candidate in observed.candidates}
        if len(ordered_candidate_ids) != len(expected) or set(ordered_candidate_ids) != expected:
            raise ValueError(
                "rank-only normalizer must return every candidate ID exactly once"
            )
        selected_id = ordered_candidate_ids[0]
        selected = next(
            candidate
            for candidate in observed.candidates
            if candidate.candidate_id == selected_id
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
        lattice_timeline: list[dict[str, object]] | None = None,
        relisten: Callable[[RelistenRequest], list[CandidateEvidence]] | None = None,
        evidence_acquire: Callable[[EvidenceAction], list[CandidateEvidence]] | None = None,
        local_rank: list[str] | None = None,
    ) -> PipelineResult:
        observed, requests, evidence_actions, diagnostics = self.observe_with_plan(
            candidates,
            source_audio_sha256=source_audio_sha256,
            segment_start_ms=segment_start_ms,
            segment_end_ms=segment_end_ms,
            token_spans=token_spans,
            lattice_timeline=lattice_timeline,
            relisten=relisten,
            evidence_acquire=evidence_acquire,
        )
        normalized = (
            self.normalize_rank_only(observed, local_rank)
            if local_rank is not None
            else None
        )
        return PipelineResult(
            observed=observed,
            normalized=normalized,
            relisten_requests=requests,
            evidence_actions=evidence_actions,
            diagnostics=diagnostics,
        )
