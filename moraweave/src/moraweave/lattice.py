from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable

from .contracts import CandidateEvidence


@dataclass(frozen=True, slots=True)
class Alternative:
    candidate_id: str
    text: str
    acoustic: float | None
    mora: float | None


@dataclass(frozen=True, slots=True)
class ContradictionIsland:
    pivot_start: int
    pivot_end: int
    pivot_text: str
    alternatives: tuple[Alternative, ...]
    start_ms: int | None = None
    end_ms: int | None = None
    kinds: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ConsensusSpan:
    start: int
    end: int
    text: str


@dataclass(frozen=True, slots=True)
class EvidenceLattice:
    pivot_candidate_id: str
    pivot_text: str
    consensus_spine: tuple[ConsensusSpan, ...]
    contradiction_islands: tuple[ContradictionIsland, ...]


def _difference_intervals(pivot: str, candidate: str) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    matcher = SequenceMatcher(a=list(pivot), b=list(candidate), autojunk=False)
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        # Insertions have zero pivot width. Anchor them to an adjacent character so that
        # they remain visible and can be mapped to an audio interval.
        if i1 == i2:
            if i1 < len(pivot):
                i2 = i1 + 1
            elif i1 > 0:
                i1 -= 1
        intervals.append((i1, i2))
    return intervals


def _merge_intervals(intervals: Iterable[tuple[int, int]], *, join_gap: int = 0) -> list[tuple[int, int]]:
    ordered = sorted((max(0, start), max(start, end)) for start, end in intervals)
    if not ordered:
        return []
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end + join_gap:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def _candidate_slice(pivot: str, candidate: str, start: int, end: int) -> str:
    """Return the candidate text aligned to one pivot interval."""

    matcher = SequenceMatcher(a=list(pivot), b=list(candidate), autojunk=False)
    pieces: list[str] = []
    for _tag, i1, i2, j1, j2 in matcher.get_opcodes():
        overlaps = max(i1, start) < min(i2, end)
        insertion_at_boundary = i1 == i2 and start <= i1 <= end
        if overlaps or insertion_at_boundary:
            pieces.append(candidate[j1:j2])
    return "".join(pieces)


def _kind(pivot_text: str, alternatives: Iterable[str]) -> tuple[str, ...]:
    combined = "".join([pivot_text, *alternatives])
    kinds: list[str] = []
    if any(char.isdigit() or char in "〇一二三四五六七八九十百千万億兆" for char in combined):
        kinds.append("number")
    if any(char in "はがをにへとでのもやかねよ" for char in combined):
        kinds.append("particle-or-functional")
    if any("一" <= char <= "龯" for char in combined):
        kinds.append("kanji-or-proper-noun")
    if not kinds:
        kinds.append("phonetic-or-punctuation")
    return tuple(kinds)


def _map_time(
    start: int,
    end: int,
    char_timeline: list[dict[str, object]] | None,
) -> tuple[int | None, int | None]:
    if not char_timeline:
        return None, None
    overlapping: list[tuple[int, int]] = []
    for row in char_timeline:
        try:
            char_start = int(row["charStart"])
            char_end = int(row["charEnd"])
            start_ms = int(row["startMs"])
            end_ms = int(row["endMs"])
        except (KeyError, TypeError, ValueError):
            continue
        if max(start, char_start) < min(end, char_end):
            overlapping.append((start_ms, end_ms))
    if not overlapping:
        return None, None
    return min(item[0] for item in overlapping), max(item[1] for item in overlapping)


def build_evidence_lattice(
    candidates: list[CandidateEvidence],
    *,
    pivot_candidate_id: str | None = None,
    char_timeline: list[dict[str, object]] | None = None,
    join_gap_chars: int = 1,
) -> EvidenceLattice:
    if not candidates:
        raise ValueError("at least one candidate is required")
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise ValueError("candidate IDs must be unique")

    if pivot_candidate_id is None:
        pivot = max(
            candidates,
            key=lambda item: (
                float("-inf") if item.acoustic is None else item.acoustic,
                item.candidate_id,
            ),
        )
    else:
        try:
            pivot = next(item for item in candidates if item.candidate_id == pivot_candidate_id)
        except StopIteration as exc:
            raise ValueError("pivot candidate is absent from the lattice") from exc

    intervals = _merge_intervals(
        (
            interval
            for candidate in candidates
            if candidate.candidate_id != pivot.candidate_id
            for interval in _difference_intervals(pivot.text, candidate.text)
        ),
        join_gap=join_gap_chars,
    )

    islands: list[ContradictionIsland] = []
    for start, end in intervals:
        alternatives = tuple(
            Alternative(
                candidate_id=candidate.candidate_id,
                text=_candidate_slice(pivot.text, candidate.text, start, end),
                acoustic=candidate.acoustic,
                mora=candidate.mora,
            )
            for candidate in candidates
        )
        start_ms, end_ms = _map_time(start, end, char_timeline)
        islands.append(
            ContradictionIsland(
                pivot_start=start,
                pivot_end=end,
                pivot_text=pivot.text[start:end],
                alternatives=alternatives,
                start_ms=start_ms,
                end_ms=end_ms,
                kinds=_kind(pivot.text[start:end], (item.text for item in alternatives)),
            )
        )

    consensus: list[ConsensusSpan] = []
    cursor = 0
    for start, end in intervals:
        if cursor < start:
            consensus.append(ConsensusSpan(cursor, start, pivot.text[cursor:start]))
        cursor = max(cursor, end)
    if cursor < len(pivot.text):
        consensus.append(ConsensusSpan(cursor, len(pivot.text), pivot.text[cursor:]))

    return EvidenceLattice(
        pivot_candidate_id=pivot.candidate_id,
        pivot_text=pivot.text,
        consensus_spine=tuple(span for span in consensus if span.text),
        contradiction_islands=tuple(islands),
    )


def islands_to_token_spans(lattice: EvidenceLattice) -> list[dict[str, object]]:
    spans: list[dict[str, object]] = []
    for island in lattice.contradiction_islands:
        if island.start_ms is None or island.end_ms is None:
            continue
        alternatives = {item.text for item in island.alternatives}
        diversity = min(1.0, max(0.0, (len(alternatives) - 1) / max(1, len(island.alternatives) - 1)))
        spans.append(
            {
                "startMs": island.start_ms,
                "endMs": island.end_ms,
                "confidence": 1.0 - diversity,
                "moraDisagreement": diversity,
                "kinds": list(island.kinds),
            }
        )
    return spans
