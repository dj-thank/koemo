from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from moraweave.adapters import DecodeRequest
from moraweave.calibrate_cli import main as calibrate_main
from moraweave.contracts import CandidateEvidence, MoraUnit
from moraweave.longform_v2 import FrontierLongformTranscriber, plan_windows, stitch_text
from moraweave.runtime_cache import (
    RuntimeCacheKey,
    RuntimeEvidenceCache,
    TeacherCacheEntry,
)
from moraweave.transcribe_v2 import write_outputs


class FakeAdapter:
    name = "fake-whisper"
    model_name = "fixture"

    def __init__(self) -> None:
        self.calls: list[DecodeRequest] = []

    def decode(self, request: DecodeRequest) -> list[CandidateEvidence]:
        self.calls.append(request)
        if request.beam_size > 5:
            return [
                CandidateEvidence(
                    "relisten",
                    "今日は三人です",
                    acoustic=0.86,
                    mora=0.82,
                    rank=1,
                    hypothesis_count=1,
                    avg_logprob=-0.08,
                    source=self.name,
                )
            ]
        return [
            CandidateEvidence(
                "a",
                "今日は三人です",
                acoustic=0.51,
                mora=0.49,
                rank=1,
                hypothesis_count=2,
                avg_logprob=-0.31,
                source=self.name,
            ),
            CandidateEvidence(
                "b",
                "今日は二人です",
                acoustic=0.50,
                mora=0.51,
                rank=2,
                hypothesis_count=2,
                avg_logprob=-0.33,
                source=self.name,
            ),
        ]


def test_runtime_cache_roundtrip_preserves_frontier_evidence() -> None:
    with tempfile.TemporaryDirectory() as directory:
        key = RuntimeCacheKey.create(
            namespace="base",
            audio_sha256="a" * 64,
            start_ms=0,
            end_ms=1000,
            adapter="fake",
            model="fixture",
            language="ja",
            beam_size=5,
            hypotheses=2,
            initial_prompt="技術会議",
            hotwords=("MoraWeave", "Qwen"),
            context="文脈",
            calibration_digest="b" * 64,
        )
        candidate = CandidateEvidence(
            "a",
            "きょうです",
            token_ids=(1, 2),
            acoustic=-0.2,
            mora=0.7,
            rank=1,
            hypothesis_count=2,
            sequence_score=-0.4,
            avg_logprob=-0.2,
            beam_confidence=0.8,
            source="fake",
            reading="キョウデス",
            mora_units=(MoraUnit("キョ"), MoraUnit("ウ")),
            metadata={"adapter": "fake", "sourceSupport": ["fake"]},
        )
        with RuntimeEvidenceCache(Path(directory) / "cache.sqlite3") as cache:
            cache.put_candidates(key, [candidate])
            restored = cache.get_candidates(key)
            assert restored == [candidate]
            assert cache.count("base") == 1


def test_cache_key_includes_context_hotwords_and_calibration() -> None:
    common = dict(
        namespace="base",
        audio_sha256="a" * 64,
        start_ms=0,
        end_ms=1000,
        adapter="fake",
        model="fixture",
        language="ja",
        beam_size=5,
        hypotheses=2,
    )
    first = RuntimeCacheKey.create(**common, context="A", hotwords=("東京",))
    second = RuntimeCacheKey.create(**common, context="B", hotwords=("東京",))
    third = RuntimeCacheKey.create(**common, context="A", hotwords=("京都",))
    fourth = RuntimeCacheKey.create(**common, context="A", hotwords=("東京",), calibration_digest="c" * 64)
    assert len({first.digest, second.digest, third.digest, fourth.digest}) == 4


def test_teacher_cache_preserves_abstention() -> None:
    with tempfile.TemporaryDirectory() as directory:
        key = RuntimeCacheKey.create(
            namespace="teacher-rank",
            audio_sha256="a" * 64,
            start_ms=0,
            end_ms=1000,
            adapter="local-teacher",
            model="qwen-local",
            language="ja",
            beam_size=5,
            hypotheses=2,
            context="candidate lattice",
        )
        entry = TeacherCacheEntry(
            probabilities={"a": 0.5, "b": 0.5},
            abstained=True,
            entropy=1.0,
            model="qwen-local",
            protocol="ollama",
        )
        with RuntimeEvidenceCache(Path(directory) / "cache.sqlite3") as cache:
            cache.put_teacher(key, entry)
            restored = cache.get_teacher(key)
            assert restored == entry
            assert restored is not None and restored.abstained is True


def test_window_plan_and_japanese_overlap_stitching() -> None:
    windows = plan_windows(60_000, window_ms=28_000, overlap_ms=1_000)
    assert [(row.start_ms, row.end_ms) for row in windows] == [
        (0, 28_000),
        (27_000, 55_000),
        (54_000, 60_000),
    ]
    assert stitch_text("今日は学校へ", "学校へ行きます。") == "今日は学校へ行きます。"
    assert stitch_text("OpenAI", "API") == "OpenAI API"


def test_longform_selective_decode_is_cached() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        audio = root / "fixture.wav"
        audio.write_bytes(b"not-real-audio-fake-adapter")
        adapter = FakeAdapter()
        with RuntimeEvidenceCache(root / "cache.sqlite3") as cache:
            transcriber = FrontierLongformTranscriber(
                adapter,
                cache=cache,
                window_ms=20_000,
                overlap_ms=1_000,
            )
            first = transcriber.transcribe(audio, duration_ms=30_000)
            first_call_count = len(adapter.calls)
            assert first_call_count > 0
            second = transcriber.transcribe(audio, duration_ms=30_000)
            assert len(adapter.calls) == first_call_count
            assert second.diagnostics["cacheHitCount"] > 0
            assert first.observed_text
            assert second.observed_text == first.observed_text


def test_output_redacts_absolute_path_by_default() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        audio = root / "fixture.wav"
        audio.write_bytes(b"fixture")
        adapter = FakeAdapter()
        result = FrontierLongformTranscriber(adapter).transcribe(audio, duration_ms=1000)
        outputs = write_outputs(result, root / "out", source_name=audio.name)
        payload = json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))
        assert payload["source_path"] == "fixture.wav"
        assert str(root) not in Path(outputs["json"]).read_text(encoding="utf-8")
        assert payload["contract"]["observedImmutable"] is True


def test_calibration_cli_writes_profile() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "calibration.jsonl"
        rows = [
            {"confidence": confidence, "correct": correct}
            for confidence, correct in zip(
                [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.45, 0.35, 0.2, 0.1],
                [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                strict=True,
            )
        ]
        source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        target = root / "profile.json"
        assert calibrate_main([str(source), "--output", str(target)]) == 0
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert payload["sampleCount"] == 10
        assert payload["profile"]["temperature"] > 0
        assert len(payload["profile"]["digest"]) == 64


def test_invalid_cache_schema_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "cache.sqlite3"
        import sqlite3

        connection = sqlite3.connect(path)
        connection.execute("CREATE TABLE cache_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("INSERT INTO cache_metadata VALUES('schema_version', '999')")
        connection.commit()
        connection.close()
        with pytest.raises(RuntimeError):
            RuntimeEvidenceCache(path)
