from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from moraweave.contracts import CandidateEvidence, NormalizedTranscript
from moraweave.evaluation import (
    cer,
    disfluency_preservation_rate,
    evaluate_transcript,
    mora_error_rate,
    number_error_rate,
    unsupported_correction_rate,
)
from moraweave.gates import GateConfig, gate_candidates
from moraweave.memory import HashedNgramMemory, TeacherCacheKey, TeacherProbabilityCache
from moraweave.mora import mora_sequence, to_katakana
from moraweave.pipeline import MoraWeavePipeline
from moraweave.rights import RightsRecord, RightsRegistry, pseudonymize_speaker
from moraweave.selective import merge_relisten_candidates, plan_relisten


def candidates() -> list[CandidateEvidence]:
    return [
        CandidateEvidence("spoken", "昨日学校を行きました", acoustic=.91, mora=.89, lexical=.38, preservation=.96, teacher=.08),
        CandidateEvidence("clean", "昨日学校に行きました", acoustic=.62, mora=.58, lexical=.95, preservation=.42, teacher=.92),
        CandidateEvidence("wrong", "昨日会社に行きました", acoustic=.31, mora=.22, lexical=.83, preservation=.21, teacher=.71),
    ]


def registry() -> RightsRegistry:
    return RightsRegistry([
        RightsRecord(
            asset_id="fixture", source_name="fixture", source_url="https://example.invalid",
            license_name="test", license_url="https://example.invalid/license",
            train="allow", derive_features="allow", redistribute_raw="deny",
            export_speaker_id="deny", attribution="fixture", reviewed_at="2026-08-28",
        )
    ])


def test_japanese_mora_normalization() -> None:
    assert to_katakana("きゃ ﾃｨ") == "キャ ティ"
    assert mora_sequence("きゃく") == ["キャ", "ク"]
    assert mora_sequence("がっこう") == ["ガ", "ッ", "コ", "ウ"]
    assert mora_sequence("スーパー") == ["ス", "ー", "パ", "ー"]


def test_acoustically_supported_error_wins_and_honeytrap_is_penalized() -> None:
    ranked = gate_candidates(candidates())
    assert ranked[0].candidate.candidate_id == "spoken"
    clean = next(item for item in ranked if item.candidate.candidate_id == "clean")
    assert clean.grammar_honeytrap_penalty > 0
    assert sum(ranked[0].gate.weights.values()) == pytest.approx(1.0)


def test_duplicate_ids_fail_closed() -> None:
    values = candidates()
    with pytest.raises(ValueError):
        gate_candidates([values[0], replace(values[1], candidate_id="spoken")])


def test_ambiguity_relistens_but_clear_candidate_skips() -> None:
    close = [
        CandidateEvidence("a", "東京です", acoustic=.51, mora=.50, lexical=.50, preservation=.50),
        CandidateEvidence("b", "東京でした", acoustic=.50, mora=.51, lexical=.51, preservation=.50),
    ]
    ranked = gate_candidates(close, GateConfig.default())
    assert ranked[0].gate.needs_relisten
    assert plan_relisten(ranked, segment_start_ms=1000, segment_end_ms=3000)

    clear = [
        CandidateEvidence("a", "東京です", acoustic=1, mora=1, lexical=1, preservation=1),
        CandidateEvidence("b", "東京でした", acoustic=0, mora=0, lexical=0, preservation=0),
    ]
    assert not gate_candidates(clear)[0].gate.needs_relisten


def test_relisten_merge_keeps_strongest_and_source_support() -> None:
    merged = merge_relisten_candidates(
        [CandidateEvidence("a", "同じ文", acoustic=.2, source="first")],
        [CandidateEvidence("b", "同じ文", acoustic=.9, source="second")],
    )
    assert [candidate.candidate_id for candidate in merged] == ["b"]
    assert merged[0].metadata["sourceSupport"] == ["first", "second"]


def test_observed_contract_and_rank_only_normalization() -> None:
    pipeline = MoraWeavePipeline()
    observed, _, _ = pipeline.observe(candidates(), source_audio_sha256="a" * 64)
    observed.verify()
    assert observed.text == "昨日学校を行きました"
    with pytest.raises(ValueError):
        replace(observed, text="昨日学校に行きました").verify()
    normalized = pipeline.normalize_rank_only(observed, ["clean", "spoken", "wrong"])
    assert normalized.text == "昨日学校に行きました"
    assert normalized.observed_evidence_sha256 == observed.evidence_sha256
    with pytest.raises(ValueError):
        NormalizedTranscript.attach(observed, text="発明文", mode="rank-only", selected_candidate_id="invented")


def test_rights_and_hashed_memory_fail_closed() -> None:
    assert registry().require("fixture", "derive_features").asset_id == "fixture"
    with pytest.raises(PermissionError):
        registry().require("fixture", "redistribute_raw")
    with pytest.raises(PermissionError):
        registry().require("unknown", "train")
    pseudonym = pseudonymize_speaker("speaker@example", b"0123456789abcdef")
    assert pseudonym == pseudonymize_speaker("speaker@example", b"0123456789abcdef")
    assert "speaker" not in pseudonym

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "memory.sqlite3"
        memory = HashedNgramMemory(path)
        report = memory.ingest(["東京で音声認識を研究します"], asset_id="fixture", registry=registry())
        assert report["uniqueHashedNgrams"] > 0
        memory.close()
        assert "東京で音声認識を研究します".encode() not in path.read_bytes()


def test_teacher_cache_is_context_keyed_and_distribution_checked() -> None:
    with tempfile.TemporaryDirectory() as directory:
        cache = TeacherProbabilityCache(Path(directory) / "teacher.sqlite3")
        key = TeacherCacheKey.create(model="local", context="文脈", candidates=[{"id": "a", "text": "候補"}], audio_digest="b" * 64)
        cache.put(key, {"a": 1.0})
        assert cache.get(key) == {"a": 1.0}
        other = TeacherCacheKey.create(model="local", context="別文脈", candidates=[{"id": "a", "text": "候補"}], audio_digest="b" * 64)
        assert cache.get(other) is None
        with pytest.raises(ValueError):
            cache.put(key, {"a": .7, "b": .7})


def test_metrics_have_correct_denominator_and_preserve_unknowns() -> None:
    assert cer("今日は3人です", "今日は2人です") == pytest.approx(1 / 7)
    assert number_error_rate("3人で1000円", "2人で1000円") == pytest.approx(.5)
    assert mora_error_rate(None, None) is None
    result = evaluate_transcript(reference="学校", observed="学校", normalized="学校")
    assert result.mler is None and result.kana_cer is None


def test_disfluency_and_unsupported_correction_metrics() -> None:
    assert disfluency_preservation_rate("えっと今日は、あの、学校へ", "今日は学校へ") == 0
    assert disfluency_preservation_rate("えっと今日は、あの、学校へ", "えっと今日は、あの、学校へ") == 1
    assert unsupported_correction_rate("学校を行きました", "学校に行きました") > 0
    assert unsupported_correction_rate("学校を行きました", "学校を行きました") == 0


def test_pipeline_output_is_serializable() -> None:
    result = MoraWeavePipeline().run(candidates())
    json.dumps(result.as_dict(), ensure_ascii=False, default=str)
    assert result.diagnostics["candidateCount"] == 3
