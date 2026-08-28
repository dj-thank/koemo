from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..rights import RightsRegistry


def _hash(namespace: str, value: str) -> str:
    return hashlib.blake2b(f"{namespace}\0{value}".encode("utf-8"), digest_size=16).hexdigest()


def character_ngrams(text: str, minimum: int = 2, maximum: int = 5) -> Iterable[str]:
    compact = "".join(str(text).split())
    for size in range(minimum, maximum + 1):
        for start in range(max(0, len(compact) - size + 1)):
            yield compact[start : start + size]


class HashedNgramMemory:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS ngrams(namespace TEXT NOT NULL, digest TEXT NOT NULL, n INTEGER NOT NULL, count INTEGER NOT NULL, PRIMARY KEY(namespace,digest))"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS builds(build_id TEXT PRIMARY KEY, asset_id TEXT NOT NULL, input_digest TEXT NOT NULL, rows_seen INTEGER NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def ingest(self, texts: Iterable[str], *, asset_id: str, registry: RightsRegistry, namespace: str = "public-ja") -> dict[str, object]:
        registry.require(asset_id, "derive_features")
        counter: Counter[tuple[str, int]] = Counter()
        input_hasher = hashlib.sha256()
        rows_seen = 0
        for text in texts:
            value = str(text).strip()
            if not value:
                continue
            rows_seen += 1
            input_hasher.update(value.encode("utf-8")); input_hasher.update(b"\n")
            for ngram in character_ngrams(value):
                counter[(_hash(namespace, ngram), len(ngram))] += 1
        input_digest = input_hasher.hexdigest()
        build_id = _hash(asset_id, input_digest)
        with self.connection:
            for (digest, size), count in counter.items():
                self.connection.execute(
                    "INSERT INTO ngrams(namespace,digest,n,count) VALUES(?,?,?,?) ON CONFLICT(namespace,digest) DO UPDATE SET count=count+excluded.count",
                    (namespace, digest, size, count),
                )
            self.connection.execute(
                "INSERT OR REPLACE INTO builds(build_id,asset_id,input_digest,rows_seen) VALUES(?,?,?,?)",
                (build_id, asset_id, input_digest, rows_seen),
            )
        return {"buildId": build_id, "assetId": asset_id, "inputDigest": input_digest, "rowsSeen": rows_seen, "uniqueHashedNgrams": len(counter)}

    def score(self, text: str, *, namespace: str = "public-ja") -> float | None:
        ngrams = list(character_ngrams(text))
        if not ngrams:
            return None
        values = []
        for ngram in ngrams:
            row = self.connection.execute(
                "SELECT count FROM ngrams WHERE namespace=? AND digest=?",
                (namespace, _hash(namespace, ngram)),
            ).fetchone()
            values.append(math.log1p(int(row[0]) if row else 0))
        return sum(values) / len(values)


@dataclass(frozen=True, slots=True)
class TeacherCacheKey:
    model: str
    context_digest: str
    candidates_digest: str
    audio_digest: str | None

    @classmethod
    def create(cls, *, model: str, context: str, candidates: list[dict[str, object]], audio_digest: str | None) -> "TeacherCacheKey":
        return cls(
            model=model,
            context_digest=hashlib.sha256(context.encode("utf-8")).hexdigest(),
            candidates_digest=hashlib.sha256(json.dumps(candidates, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
            audio_digest=audio_digest,
        )

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(dataclasses.asdict(self), sort_keys=True).encode("utf-8")).hexdigest()


class TeacherProbabilityCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS teacher_cache(key_digest TEXT PRIMARY KEY, model TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        self.connection.commit()

    def get(self, key: TeacherCacheKey) -> dict[str, float] | None:
        row = self.connection.execute("SELECT payload_json FROM teacher_cache WHERE key_digest=?", (key.digest(),)).fetchone()
        if row is None:
            return None
        return {str(name): float(value) for name, value in json.loads(row[0]).items()}

    def put(self, key: TeacherCacheKey, probabilities: dict[str, float]) -> None:
        if any(not 0.0 <= value <= 1.0 for value in probabilities.values()):
            raise ValueError("teacher probabilities must be in [0,1]")
        if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-4):
            raise ValueError("teacher probabilities must sum to one")
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO teacher_cache(key_digest,model,payload_json) VALUES(?,?,?)",
                (key.digest(), key.model, json.dumps(probabilities, sort_keys=True)),
            )


__all__ = ["HashedNgramMemory", "TeacherCacheKey", "TeacherProbabilityCache", "character_ngrams"]
