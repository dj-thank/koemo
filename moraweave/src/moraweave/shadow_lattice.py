from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable, Literal, Mapping, Sequence

from .contracts import CandidateEvidence
from .mora import mora_sequence, to_katakana

AlignmentLevel = Literal["surface", "mora"]
_PARTICLES = frozenset("はがをにへとでのもやかねよぞさなってからまでよりしかば")
_NUMBER_RE = re.compile(r"[0-9〇一二三四五六七八九十百千万億兆]")
_DATE_TIME_RE = re.compile(
    r"(?:\d{1,4}[/-]\d{1,2}(?:[/-]\d{1,2})?|\d{1,2}:\d{2}|[年月日時分秒])"
)
_CURRENCY_RE = re.compile(r"[¥￥$€£]|円|ドル|ユーロ|ポンド")
_LATIN_RE = re.compile(r"[A-Za-z]{2,}")


@dataclass(frozen=True, slots=True)
class ShadowAlternative:
    candidate_id: str
    units: tuple[str, ...]
    surface_text: str
    posterior: float
    source: str


@dataclass(frozen=True, slots=True)
class ShadowIsland:
    start: int
    end: int
    pivot_units: tuple[str, ...]
    alternatives: tuple[ShadowAlternative, ...]
    kinds: tuple[str, ...]
    posterior_ambiguity: float
    criticality: float
    expected_information_gain: float
    start_ms: int | None = None
    end_ms: int | None = None


@dataclass(frozen=True, slots=True)
class LockedConsensus:
    start: int
    end: int
    units: tuple[str, ...]
    support: float = 1.0


@dataclass(frozen=True, slots=True)
class DualEvidenceLattice:
    pivot_candidate_id: str
    alignment_level: AlignmentLevel
    pivot_units: tuple[str, ...]
    locked_consensus: tuple[LockedConsensus, ...]
    contradiction_islands: tuple[ShadowIsland, ...]


def _reading(candidate: CandidateEvidence) -> str | None:
    if candidate.reading:
        return candidate.reading
    metadata_reading = candidate.metadata.get("reading")
    return str(metadata_reading) if metadata_reading else None


def _has_mora_shadow(candidate: CandidateEvidence) -> bool:
    return bool(candidate.mora_units or _reading(candidate))


def _surface_units(text: str) -> tuple[str, ...]:
    value = unicodedata.normalize("NFKC", text)
    return tuple(character for character in value if not character.isspace())


def _mora_units(candidate: CandidateEvidence) -> tuple[str, ...]:
    if candidate.mora_units:
        return tuple(unit.kana for unit in candidate.mora_units)
    reading = _reading(candidate)
    return tuple(mora_sequence(reading)) if reading else ()


def _candidate_units(candidate: CandidateEvidence, level: AlignmentLevel) -> tuple[str, ...]:
    return _mora_units(candidate) if level == "mora" else _surface_units(candidate.text)


def _difference_intervals(pivot: Sequence[str], candidate: Sequence[str]) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    matcher = SequenceMatcher(a=list(pivot), b=list(candidate), autojunk=False)
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if i1 == i2:
            if i1 < len(pivot):
                i2 = i1 + 1
            elif i1 > 0:
                i1 -= 1
        intervals.append((i1, i2))
    return intervals


def _merge_intervals(intervals: Iterable[tuple[int, int]], *, join_gap: int) -> list[tuple[int, int]]:
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


def _candidate_slice(
    pivot: Sequence[str], candidate: Sequence[str], start: int, end: int
) -> tuple[str, ...]:
    matcher = SequenceMatcher(a=list(pivot), b=list(candidate), autojunk=False)
    pieces: list[str] = []
    for _tag, i1, i2, j1, j2 in matcher.get_opcodes():
        overlaps = max(i1, start) < min(i2, end)
        insertion_at_boundary = i1 == i2 and start <= i1 <= end
        if overlaps or insertion_at_boundary:
            pieces.extend(candidate[j1:j2])
    return tuple(pieces)


def _kinds(pivot_units: Sequence[str], alternatives: Iterable[Sequence[str]]) -> tuple[str, ...]:
    texts = ["".join(pivot_units), *("".join(value) for value in alternatives)]
    combined = "".join(texts)
    kinds: list[str] = []
    if _NUMBER_RE.search(combined):
        kinds.append("number")
    if _DATE_TIME_RE.search(combined):
        kinds.append("date-or-time")
    if _CURRENCY_RE.search(combined):
        kinds.append("currency")
    if any(len(text) <= 3 and any(character in _PARTICLES for character in text) for text in texts):
        kinds.append("particle-or-functional")
    if _LATIN_RE.search(combined):
        kinds.append("latin-acronym-or-term")
    if any("一" <= character <= "龯" for character in combined):
        kinds.append("kanji-or-proper-noun")
    if any(character in "ンッー" for character in to_katakana(combined)):
        kinds.append("special-mora")
    if not kinds:
        kinds.append("phonetic-or-punctuation")
    return tuple(dict.fromkeys(kinds))


def _criticality(kinds: Sequence[str]) -> float:
    weights = {
        "number": 1.00,
        "date-or-time": 1.00,
        "currency": 1.00,
        "kanji-or-proper-noun": 0.92,
        "latin-acronym-or-term": 0.88,
        "special-mora": 0.82,
        "particle-or-functional": 0.72,
        "phonetic-or-punctuation": 0.45,
    }
    return max(weights.get(kind, 0.45) for kind in kinds)


def _ambiguity(alternatives: Sequence[ShadowAlternative]) -> float:
    mass_by_units: dict[tuple[str, ...], float] = {}
    for alternative in alternatives:
        mass_by_units[alternative.units] = mass_by_units.get(alternative.units, 0.0) + alternative.posterior
    total = sum(mass_by_units.values())
    if total <= 0 or len(mass_by_units) <= 1:
        return 0.0
    normalized = [mass / total for mass in mass_by_units.values()]
    entropy = -sum(probability * math.log(probability + 1e-12) for probability in normalized)
    return min(1.0, entropy / math.log(len(normalized)))


def _map_time(
    start: int,
    end: int,
    timeline: list[dict[str, object]] | None,
    *,
    level: AlignmentLevel,
) -> tuple[int | None, int | None]:
    if not timeline:
        return None, None
    overlapping: list[tuple[int, int]] = []
    for row in timeline:
        try:
            if level == "mora":
                raw_start = row.get("moraStart", row.get("unitStart", row.get("index")))
                if raw_start is None:
                    continue
                unit_start = int(raw_start)
                unit_end = int(row.get("moraEnd", row.get("unitEnd", unit_start + 1)))
            else:
                unit_start = int(row["charStart"])
                unit_end = int(row["charEnd"])
            start_ms = int(row["startMs"])
            end_ms = int(row["endMs"])
        except (KeyError, TypeError, ValueError):
            continue
        if max(start, unit_start) < min(end, unit_end):
            overlapping.append((start_ms, end_ms))
    if not overlapping:
        return None, None
    return min(value[0] for value in overlapping), max(value[1] for value in overlapping)


def build_dual_evidence_lattice(
    candidates: list[CandidateEvidence],
    *,
    posterior: Mapping[str, float] | None = None,
    pivot_candidate_id: str | None = None,
    timeline: list[dict[str, object]] | None = None,
    join_gap_units: int = 1,
) -> DualEvidenceLattice:
    """Build surface or mora-shadow contradiction islands."""

    if not candidates:
        raise ValueError("at least one candidate is required")
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise ValueError("candidate IDs must be unique")
    level: AlignmentLevel = "mora" if all(_has_mora_shadow(candidate) for candidate in candidates) else "surface"
    if pivot_candidate_id is None:
        pivot = max(
            candidates,
            key=lambda candidate: (
                float("-inf") if candidate.acoustic is None else candidate.acoustic,
                candidate.candidate_id,
            ),
        )
    else:
        try:
            pivot = next(candidate for candidate in candidates if candidate.candidate_id == pivot_candidate_id)
        except StopIteration as exc:
            raise ValueError("pivot candidate is absent") from exc
    units_by_id = {candidate.candidate_id: _candidate_units(candidate, level) for candidate in candidates}
    pivot_units = units_by_id[pivot.candidate_id]
    intervals = _merge_intervals(
        (
            interval
            for candidate in candidates
            if candidate.candidate_id != pivot.candidate_id
            for interval in _difference_intervals(pivot_units, units_by_id[candidate.candidate_id])
        ),
        join_gap=join_gap_units,
    )
    if posterior is None:
        probability = 1.0 / len(candidates)
        posterior = {candidate.candidate_id: probability for candidate in candidates}
    else:
        total = sum(max(0.0, float(posterior.get(candidate.candidate_id, 0.0))) for candidate in candidates)
        if total <= 0:
            raise ValueError("posterior mass must be positive")
        posterior = {
            candidate.candidate_id: max(0.0, float(posterior.get(candidate.candidate_id, 0.0))) / total
            for candidate in candidates
        }
    islands: list[ShadowIsland] = []
    for start, end in intervals:
        alternatives = tuple(
            ShadowAlternative(
                candidate_id=candidate.candidate_id,
                units=_candidate_slice(pivot_units, units_by_id[candidate.candidate_id], start, end),
                surface_text=candidate.text,
                posterior=float(posterior[candidate.candidate_id]),
                source=candidate.evidence_source,
            )
            for candidate in candidates
        )
        kinds = _kinds(pivot_units[start:end], (alternative.units for alternative in alternatives))
        ambiguity = _ambiguity(alternatives)
        criticality = _criticality(kinds)
        start_ms, end_ms = _map_time(start, end, timeline, level=level)
        islands.append(
            ShadowIsland(
                start=start,
                end=end,
                pivot_units=tuple(pivot_units[start:end]),
                alternatives=alternatives,
                kinds=kinds,
                posterior_ambiguity=ambiguity,
                criticality=criticality,
                expected_information_gain=ambiguity * criticality,
                start_ms=start_ms,
                end_ms=end_ms,
            )
        )
    consensus: list[LockedConsensus] = []
    cursor = 0
    for start, end in intervals:
        if cursor < start:
            consensus.append(LockedConsensus(cursor, start, tuple(pivot_units[cursor:start])))
        cursor = max(cursor, end)
    if cursor < len(pivot_units):
        consensus.append(LockedConsensus(cursor, len(pivot_units), tuple(pivot_units[cursor:])))
    return DualEvidenceLattice(
        pivot_candidate_id=pivot.candidate_id,
        alignment_level=level,
        pivot_units=tuple(pivot_units),
        locked_consensus=tuple(span for span in consensus if span.units),
        contradiction_islands=tuple(islands),
    )


def lattice_token_spans(lattice: DualEvidenceLattice) -> list[dict[str, object]]:
    spans: list[dict[str, object]] = []
    for island in lattice.contradiction_islands:
        if island.start_ms is None or island.end_ms is None:
            continue
        spans.append(
            {
                "startMs": island.start_ms,
                "endMs": island.end_ms,
                "confidence": 1.0 - island.posterior_ambiguity,
                "moraDisagreement": island.posterior_ambiguity if lattice.alignment_level == "mora" else 0.0,
                "posteriorAmbiguity": island.posterior_ambiguity,
                "criticality": island.criticality,
                "expectedInformationGain": island.expected_information_gain,
                "kinds": list(island.kinds),
                "alignmentLevel": lattice.alignment_level,
            }
        )
    return spans
