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
    def decode_span(self, audio_path: str, request: RelistenRequest) -> list[CandidateEvidence]: ...


def _merge_spans(spans: list[TimeSpan], join_gap_ms: int = 240) -> list[TimeSpan]:
    if not spans:
        return []
    ordered = sorted(spans, key=lambda item: (item.start_ms, item.end_ms))
    merged: list[TimeSpan] = [ordered[0]]
    for span in ordered[1:]:
        previous = merged[-1]
        if span.start_ms <= previous.end_ms + join_gap_ms:
            merged[-1] = TimeSpan(
                start_ms=previous.start_ms,
                end_ms=max(previous.end_ms, span.end_ms),
                reasons=tuple(sorted(set(previous.reasons + span.reasons))),
                priority=max(previous.priority, span.priority),
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
            confidence = item.get("confidence")
            disagreement = item.get("moraDisagreement")
            try:
                confidence_value = float(confidence) if confidence is not None else 1.0
                disagreement_value = float(disagreement) if disagreement is not None else 0.0
                start = int(item["startMs"])
                end = int(item["endMs"])
            except (KeyError, TypeError, ValueError):
                continue
            local_priority = max(1.0 - confidence_value, disagreement_value)
            if local_priority >= 0.35:
                spans.append(
                    TimeSpan(
                        start_ms=max(segment_start_ms, start - 120),
                        end_ms=min(segment_end_ms, end + 120),
                        reasons=tuple(sorted(set(gate.reasons + ("local-token-uncertainty",)))),
                        priority=max(base_priority, local_priority),
                    )
                )

    if not spans:
        spans.append(
            TimeSpan(
                start_ms=segment_start_ms,
                end_ms=segment_end_ms,
                reasons=gate.reasons or ("global-candidate-ambiguity",),
                priority=base_priority,
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
    return sorted(selected, key=lambda item: item.span.start_ms)


def merge_relisten_candidates(
    original: list[CandidateEvidence],
    additional: list[CandidateEvidence],
) -> list[CandidateEvidence]:
    """Deduplicate by text while retaining the strongest acoustic evidence."""

    by_text: dict[str, CandidateEvidence] = {}
    for candidate in [*original, *additional]:
        current = by_text.get(candidate.text)
        if current is None:
            by_text[candidate.text] = candidate
            continue
        current_score = float("-inf") if current.acoustic is None else current.acoustic
        new_score = float("-inf") if candidate.acoustic is None else candidate.acoustic
        if new_score > current_score:
            by_text[candidate.text] = candidate
    return sorted(by_text.values(), key=lambda item: item.candidate_id)
