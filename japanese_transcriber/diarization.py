from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .types import Segment


@dataclass(frozen=True, slots=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str


def parse_rttm(path: str | Path) -> list[SpeakerTurn]:
    turns: list[SpeakerTurn] = []
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8 or parts[0].upper() != "SPEAKER":
            raise ValueError(f"invalid RTTM line {line_number}: {raw_line}")
        start = float(parts[3])
        duration = float(parts[4])
        if start < 0 or duration <= 0:
            raise ValueError(f"invalid RTTM time on line {line_number}")
        turns.append(SpeakerTurn(start=start, end=start + duration, speaker=parts[7]))
    return sorted(turns, key=lambda item: (item.start, item.end, item.speaker))


def _overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def choose_speaker(start: float | None, end: float | None, turns: list[SpeakerTurn]) -> str | None:
    if start is None or end is None or end < start:
        return None
    best: tuple[float, str] | None = None
    midpoint = (start + end) / 2
    for turn in turns:
        overlap = _overlap(start, end, turn.start, turn.end)
        if overlap > 0 and (best is None or overlap > best[0]):
            best = (overlap, turn.speaker)
    if best is not None:
        return best[1]
    containing = [turn for turn in turns if turn.start <= midpoint <= turn.end]
    return containing[0].speaker if containing else None


def assign_speakers(segments: list[Segment], turns: list[SpeakerTurn]) -> list[Segment]:
    for segment in segments:
        for word in segment.words:
            word.speaker = choose_speaker(word.start, word.end, turns)

        duration_by_speaker: dict[str, float] = {}
        for word in segment.words:
            if word.speaker is None or word.start is None or word.end is None:
                continue
            duration_by_speaker[word.speaker] = duration_by_speaker.get(word.speaker, 0.0) + max(0.0, word.end - word.start)

        if duration_by_speaker:
            segment.speaker = max(duration_by_speaker.items(), key=lambda item: (item[1], item[0]))[0]
        else:
            segment.speaker = choose_speaker(segment.start, segment.end, turns)
    return segments


def relabel_speakers(segments: list[Segment]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for segment in segments:
        candidates = [segment.speaker, *(word.speaker for word in segment.words)]
        for speaker in candidates:
            if speaker is not None and speaker not in mapping:
                mapping[speaker] = f"話者{len(mapping) + 1}"
    for segment in segments:
        if segment.speaker in mapping:
            segment.speaker = mapping[segment.speaker]
        for word in segment.words:
            if word.speaker in mapping:
                word.speaker = mapping[word.speaker]
    return mapping
