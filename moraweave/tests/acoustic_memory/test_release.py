from __future__ import annotations

import tempfile
from pathlib import Path

from moraweave.acoustic_memory import (
    AcousticCacheKey,
    QuerySelectedAcousticMemory,
    schedule_relisten,
)
from moraweave.contracts import CandidateEvidence
from moraweave.selective import RelistenRequest, TimeSpan


def _key(prompt: str = "") -> AcousticCacheKey:
    return AcousticCacheKey.create(
        audio_sha256="a" * 64,
        start_ms=1000,
        end_ms=2400,
        adapter="faster-whisper",
        model="large-v3-turbo",
        beam_size=12,
        hypotheses=8,
        initial_prompt=prompt,
        hotwords=("森脇渉太", "MoraWeave"),
    )


def test_acoustic_memory_roundtrip_and_context_keying() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "acoustic.sqlite3"
        memory = QuerySelectedAcousticMemory(path)
        candidates = [CandidateEvidence("a", "東京です", acoustic=.9)]
        memory.put(_key("会議"), candidates)
        assert memory.get(_key("会議")) == candidates
        assert memory.get(_key("授業")) is None
        assert memory.stats() == {"entries": 1, "hits": 1}
        memory.close()
        assert b"RIFF" not in path.read_bytes()


def test_cache_rejects_duplicate_candidate_ids() -> None:
    with tempfile.TemporaryDirectory() as directory:
        memory = QuerySelectedAcousticMemory(Path(directory) / "cache.sqlite3")
        try:
            memory.put(
                _key(),
                [
                    CandidateEvidence("same", "東京です"),
                    CandidateEvidence("same", "東京でした"),
                ],
            )
        except ValueError:
            pass
        else:
            raise AssertionError("duplicate cache candidate IDs must fail")


def test_scheduler_prioritizes_short_high-risk_island_within_budget() -> None:
    requests = [
        RelistenRequest(TimeSpan(0, 9000, ("global",), .7)),
        RelistenRequest(TimeSpan(10000, 11200, ("kind:number",), .65)),
        RelistenRequest(TimeSpan(12000, 14000, ("kind:kanji-or-proper-noun",), .7)),
    ]
    selected = schedule_relisten(requests, max_total_ms=3500)
    assert [item.request.span.start_ms for item in selected] == [10000, 12000]
    assert sum(item.cost_ms for item in selected) <= 3500
