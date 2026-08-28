from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Word:
    index: int
    text: str
    start: float | None
    end: float | None
    probability: float | None
    speaker: str | None = None
    reading: str | None = None
    mora: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Segment:
    id: str
    index: int
    start: float
    end: float
    text: str
    seek: int | None = None
    temperature: float | None = None
    avg_logprob: float | None = None
    compression_ratio: float | None = None
    no_speech_prob: float | None = None
    speaker: str | None = None
    words: list[Word] = field(default_factory=list)
    uncertainty_reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EngineResult:
    engine: dict[str, Any]
    language: dict[str, Any]
    duration: dict[str, Any]
    segments: list[Segment]


def to_dict(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, list):
        return [to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: to_dict(item) for key, item in value.items()}
    return value
