from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .contracts import CandidateEvidence, RankedCandidate


@dataclass(frozen=True, slots=True)
class TimeSpan:
    start_ms: int
    end_ms: int
    reasons: tuple[str, ...]
    priority: float

    def __post_init__(self) -> None:
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("invalid re-listening span")


@dataclass(frozen=True, slots=True)
class RelistenRequest:
    span: TimeSpan
    beam_size: int = 12
    hypotheses: int = 8
    temperature: float = 0.0
    context_before_ms: int = 600
    context_after_ms: int = 600


class SpanDecoder(Protocol):
    def decode_span(
        self, audio_path: str, request: RelistenRequest
    ) -> list[CandidateEvidence]: ...


def _merge_spans(
    spans: list[TimeSpan], join_gap_ms: int = 240
) -> list[TimeSpan]:
    if not spans:
        return []
    ordered = sorted(spans, key=lambda item: (item.start_ms, item.end_ms))
    merged = [ordered[0]]
    for span in ordered[1:]:
        previous = merged[-1]
        if span.start_ms <= previous.end_ms + join_gap_ms:
            merged[-1] = TimeSpan(
                previous.start_ms,
                max(previous.end_ms, span.end_ms),
                tuple(sorted(set(previous.reasons + span.reasons))),
                max(previous.priority, span.priority),
            )
        else:
            merged.append(span)
    return merged


def plan_relisten(
    ranked: list[RankedCandidate],
    *,
    segment_start_ms: int,
    segment_end_ms: int,
    token_spans: list[dict[str, object]] | None = None,
    max_total_ms: int = 12_000,
) -> list[RelistenRequest]:
    if not ranked:
        return []
    gate = ranked[0].gate
    if not gate.needs_relisten:
        return []
    base_priority = min(1.0, gate.entropy * 0.55 + gate.disagreement * 0.45)
    spans: list[TimeSpan] = []
    if token_spans:
        for item in token_spans:
            try:
                confidence_value = float(item.get("confidence", 1.0))
                disagreement_value = float(item.get("moraDisagreement", 0.0))
                local_ambiguity = float(item.get("posteriorAmbiguity", 0.0))
                criticality = float(item.get("criticality", 0.5))
                start = int(item["startMs"])
                end = int(item["endMs"])
            except (KeyError, TypeError, ValueError):
                continue
            local_priority = max(
                1 - confidence_value,
                disagreement_value,
                local_ambiguity,
            ) * (0.65 + 0.35 * criticality)
            if local_priority >= 0.30:
                spans.append(
                    TimeSpan(
                        max(segment_start_ms, start - 120),
                        min(segment_end_ms, end + 120),
                        tuple(
                            sorted(
                                set(
                                    gate.reasons
                                    + ("local-token-uncertainty",)
                                )
                            )
                        ),
                        max(base_priority, local_priority),
                    )
                )
    if not spans:
        spans.append(
            TimeSpan(
                segment_start_ms,
                segment_end_ms,
                gate.reasons or ("global-candidate-ambiguity",),
                base_priority,
            )
        )
    merged = sorted(_merge_spans(spans), key=lambda item: item.priority, reverse=True)
    selected: list[RelistenRequest] = []
    used = 0
    for span in merged:
        duration = span.end_ms - span.start_ms
        if selected and used + duration > max_total_ms:
            continue
        selected.append(RelistenRequest(span=span))
        used += duration
        if used >= max_total_ms:
            break
    return sorted(selected, key=lambda request: request.span.start_ms)


def merge_relisten_candidates(
    original: list[CandidateEvidence],
    additional: list[CandidateEvidence],
) -> list[CandidateEvidence]:
    """Deduplicate identical text while retaining independent source support."""

    from dataclasses import replace

    grouped: dict[str, list[CandidateEvidence]] = {}
    for candidate in [*original, *additional]:
        grouped.setdefault(candidate.text, []).append(candidate)

    output: list[CandidateEvidence] = []
    for group in grouped.values():
        strongest = max(
            group,
            key=lambda candidate: (
                float("-inf")
                if candidate.acoustic is None
                else candidate.acoustic,
                candidate.candidate_id,
            ),
        )
        sources = {
            source
            for candidate in group
            for source in (
                candidate.evidence_source,
                *(
                    str(value)
                    for value in candidate.metadata.get("sourceSupport", [])
                    if str(value)
                ),
            )
            if source
        }
        metadata = dict(strongest.metadata)
        metadata["sourceSupport"] = sorted(sources)
        output.append(
            replace(
                strongest,
                metadata=metadata,
                mora=(
                    strongest.mora
                    if strongest.mora is not None
                    else next(
                        (candidate.mora for candidate in group if candidate.mora is not None),
                        None,
                    )
                ),
                lexical=(
                    strongest.lexical
                    if strongest.lexical is not None
                    else next(
                        (candidate.lexical for candidate in group if candidate.lexical is not None),
                        None,
                    )
                ),
                preservation=(
                    strongest.preservation
                    if strongest.preservation is not None
                    else next(
                        (
                            candidate.preservation
                            for candidate in group
                            if candidate.preservation is not None
                        ),
                        None,
                    )
                ),
            )
        )
    return sorted(output, key=lambda candidate: candidate.candidate_id)
