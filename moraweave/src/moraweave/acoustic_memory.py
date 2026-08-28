from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .contracts import CandidateEvidence, MoraUnit
from .selective import RelistenRequest


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class AcousticCacheKey:
    audio_sha256: str
    start_ms: int
    end_ms: int
    adapter: str
    model: str
    beam_size: int
    hypotheses: int
    language: str
    prompt_digest: str
    hotwords_digest: str

    @classmethod
    def create(
        cls,
        *,
        audio_sha256: str,
        start_ms: int,
        end_ms: int,
        adapter: str,
        model: str,
        beam_size: int,
        hypotheses: int,
        language: str = "ja",
        initial_prompt: str | None = None,
        hotwords: Iterable[str] = (),
    ) -> "AcousticCacheKey":
        if len(audio_sha256) != 64:
            raise ValueError("audio SHA-256 must contain 64 hexadecimal characters")
        if start_ms < 0 or end_ms <= start_ms:
            raise ValueError("invalid acoustic cache span")
        if beam_size < 1 or hypotheses < 1:
            raise ValueError("beam and hypothesis counts must be positive")
        return cls(
            audio_sha256=audio_sha256,
            start_ms=start_ms,
            end_ms=end_ms,
            adapter=adapter,
            model=model,
            beam_size=beam_size,
            hypotheses=hypotheses,
            language=language,
            prompt_digest=_digest(initial_prompt or ""),
            hotwords_digest=_digest(sorted(set(str(item) for item in hotwords))),
        )

    def digest(self) -> str:
        return _digest(dataclasses.asdict(self))


def _candidate_to_dict(candidate: CandidateEvidence) -> dict[str, object]:
    return dataclasses.asdict(candidate)


def _candidate_from_dict(row: dict[str, object]) -> CandidateEvidence:
    mora_units = tuple(
        MoraUnit(
            kana=str(item["kana"]),
            start_ms=item.get("start_ms"),  # type: ignore[arg-type]
            end_ms=item.get("end_ms"),  # type: ignore[arg-type]
            confidence=item.get("confidence"),  # type: ignore[arg-type]
            phones=tuple(str(value) for value in item.get("phones", [])),
            kind=str(item.get("kind", "regular")),  # type: ignore[arg-type]
        )
        for item in row.get("mora_units", [])  # type: ignore[union-attr]
    )
    return CandidateEvidence(
        candidate_id=str(row["candidate_id"]),
        text=str(row["text"]),
        token_ids=tuple(int(value) for value in row.get("token_ids", [])),  # type: ignore[union-attr]
        acoustic=_optional_float(row.get("acoustic")),
        mora=_optional_float(row.get("mora")),
        lexical=_optional_float(row.get("lexical")),
        preservation=_optional_float(row.get("preservation")),
        teacher=_optional_float(row.get("teacher")),
        mora_units=mora_units,
        metadata=dict(row.get("metadata", {})),  # type: ignore[arg-type]
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


class QuerySelectedAcousticMemory:
    """Cache selected span-decoding results without storing waveform bytes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS acoustic_cache (
              key_digest TEXT PRIMARY KEY,
              key_json TEXT NOT NULL,
              candidates_json TEXT NOT NULL,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              last_accessed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              hit_count INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def get(self, key: AcousticCacheKey) -> list[CandidateEvidence] | None:
        row = self.connection.execute(
            "SELECT candidates_json FROM acoustic_cache WHERE key_digest = ?",
            (key.digest(),),
        ).fetchone()
        if row is None:
            return None
        with self.connection:
            self.connection.execute(
                "UPDATE acoustic_cache SET hit_count = hit_count + 1, last_accessed_at = CURRENT_TIMESTAMP WHERE key_digest = ?",
                (key.digest(),),
            )
        payload = json.loads(row[0])
        return [_candidate_from_dict(item) for item in payload]

    def put(self, key: AcousticCacheKey, candidates: list[CandidateEvidence]) -> None:
        if not candidates:
            raise ValueError("acoustic cache refuses empty candidate lists")
        ids = [candidate.candidate_id for candidate in candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("cached candidate IDs must be unique")
        with self.connection:
            self.connection.execute(
                """
                INSERT OR REPLACE INTO acoustic_cache(
                  key_digest, key_json, candidates_json, created_at, last_accessed_at, hit_count
                ) VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 0)
                """,
                (
                    key.digest(),
                    json.dumps(dataclasses.asdict(key), sort_keys=True),
                    json.dumps([_candidate_to_dict(item) for item in candidates], ensure_ascii=False, sort_keys=True),
                ),
            )

    def stats(self) -> dict[str, int]:
        row = self.connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(hit_count), 0) FROM acoustic_cache"
        ).fetchone()
        return {"entries": int(row[0]), "hits": int(row[1])}


@dataclass(frozen=True, slots=True)
class ScheduledRelisten:
    request: RelistenRequest
    expected_information_gain: float
    cost_ms: int
    utility_per_second: float


_KIND_WEIGHT = {
    "number": 1.45,
    "kanji-or-proper-noun": 1.35,
    "particle-or-functional": 1.15,
    "phonetic-or-punctuation": 0.85,
}


def schedule_relisten(
    requests: list[RelistenRequest],
    *,
    max_total_ms: int = 12_000,
    minimum_gain: float = 0.08,
) -> list[ScheduledRelisten]:
    """Allocate decoding budget by estimated information gain per second.

    Priority combines the request's uncertainty priority with semantic risk tags. Long
    spans must justify their cost; number and proper-noun islands receive more weight.
    """

    if max_total_ms <= 0:
        return []
    ranked: list[ScheduledRelisten] = []
    for request in requests:
        duration = request.span.end_ms - request.span.start_ms
        if duration <= 0:
            continue
        kinds = {
            reason.split("kind:", 1)[1]
            for reason in request.span.reasons
            if reason.startswith("kind:")
        }
        kind_weight = max((_KIND_WEIGHT.get(kind, 1.0) for kind in kinds), default=1.0)
        gain = min(1.0, max(0.0, request.span.priority)) * kind_weight
        gain = min(1.5, gain)
        utility = gain / max(0.001, duration / 1000)
        if gain >= minimum_gain:
            ranked.append(
                ScheduledRelisten(
                    request=request,
                    expected_information_gain=gain,
                    cost_ms=duration,
                    utility_per_second=utility,
                )
            )

    ranked.sort(
        key=lambda item: (
            -item.utility_per_second,
            -item.expected_information_gain,
            item.request.span.start_ms,
        )
    )
    selected: list[ScheduledRelisten] = []
    used = 0
    for item in ranked:
        if used + item.cost_ms > max_total_ms:
            continue
        selected.append(item)
        used += item.cost_ms
    return sorted(selected, key=lambda item: item.request.span.start_ms)
