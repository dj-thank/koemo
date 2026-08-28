from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from moraweave.contracts import CandidateEvidence, NormalizedTranscript, ObservedTranscript
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
        CandidateEvidence("spoken", "昨日学校を行きました", acoustic=0.91, mora=0.89, lexical=0.38, preservation=0.96, teacher=0.08),
        CandidateEvidence("clean", "昨日学校に行きました", acoustic=0.62, mora=0.58, lexical=0.95, preservation=0.42, teacher=0.92),
        CandidateEvidence("wrong", "昨日会社に行きました", acoustic=0.31, mora=0.22, lexical=0.83, preservation=0.21, teacher=0.71),
    ]


def allow_registry() -> RightsRegistry:
    return RightsRegistry([
        RightsRecord(
            asset_id="fixture",
            source_name="fixture",
            source_url="https://example.invalid",
            license_name="test",
            license_url="https://example.invalid/license",
            train="allow",
            derive_features="allow",
            redistribute_raw="deny",
            export_speaker_id="deny",
            attribution="fixture",
            reviewed_at="2026-08-28",
        )
    ])


def test_nfkc_and_katakana_normalization() -> None:
    assert to_katakana("きゃ ﾃｨ") == "キャ ティ"


def test_mora_compounds_and_special_morae() -> None:
    assert mora_sequence("きゃく") == ["キャ", "ク"]
    assert mora_sequence("がっこう") == ["ガ", "ッ", "コ", "ウ"]
    assert mora_sequence("スーパー") == ["ス", "ー", "パ", "ー"]
    assert mora_sequence("しんぶん") == ["シ", "ン", "ブ", "ン"]


def test_gate_preserves_acoustically_supported_error() -> None:
    ranked = gate_candidates(candidates())
    assert ranked[0].candidate.candidate_id == "spoken"


def test_grammar_honeytrap_penalizes_clean_but_unsupported_candidate() -> None:
    ranked = gate_candidates(candidates())
    clean = next(item for item in ranked if item.candidate.candidate_id == "clean")
    spoken = next(item for item in ranked if item.candidate.candidate_id == "spoken")
    assert clean.grammar_honeytrap_penalty > spoken.grammar_honeytrap_penalty


def test_gate_weights_sum_to_one() -> None:
    ranked = gate_candidates(candidates())
    assert sum(ranked[0].gate.weights.values()) == pytest.approx(1.0)


def test_duplicate_candidate_ids_fail_closed() -> None:
    with pytest.raises(ValueError):
        gate_candidates([candidates()[0], replace(candidates()[1], candidate_id="spoken")])


def test_close_candidates_trigger_relisten() -> None:
    close = [
        CandidateEvidence("a", "東京です", acoustic=0.51, mora=0.50, lexical=0.50, preservation=0.50),
        CandidateEvidence("b", "東京でした", acoustic=0.50, mora=0.51, lexical=0.51, preservation=0.50),
    ]
    ranked = gate_candidates(close, GateConfig.default())
    assert ranked[0].gate.needs_relisten
    requests = plan_relisten(ranked, segment_start_ms=1000, segment_end_ms=3000)
    assert len(requests) == 1
    assert requests[0].span.start_ms == 1000


def test_confident_candidates_skip_relisten() -> None:
    strong = [
        CandidateEvidence("a", "東京です", acoustic=1.0, mora=1.0, lexical=1.0, preservation=1.0),
        CandidateEvidence("b", "東京でした", acoustic=0.0, mora=0.0, lexical=0.0, preservation=0.0),
    ]
    ranked = gate_candidates(strong)
    assert not ranked[0].gate.needs_relisten
    assert plan_relisten(ranked, segment_start_ms=0, segment_end_ms=1000) == []


def test_relisten_candidate_merge_keeps_stronger_acoustic_copy() -> None:
    original = [CandidateEvidence("a", "同じ文", acoustic=0.2)]
    additional = [CandidateEvidence("a2", "同じ文", acoustic=0.9)]
    merged = merge_relisten_candidates(original, additional)
    assert len(merged) == 1
    assert merged[0].candidate_id == "a2"


def test_observed_transcript_verifies() -> None:
    result = MoraWeavePipeline().run(candidates(), source_audio_sha256="a" * 64)
    result.observed.verify()
    assert result.observed.text == "昨日学校を行きました"


def test_observed_tampering_is_detected() -> None:
    result = MoraWeavePipeline().run(candidates())
    tampered = replace(result.observed, text="昨日学校に行きました")
    with pytest.raises(ValueError):
        tampered.verify()


def test_rank_only_normalizer_must_return_exact_candidate_permutation() -> None:
    pipeline = MoraWeavePipeline()
    observed, _, _ = pipeline.observe(candidates())
    normalized = pipeline.normalize_rank_only(observed, ["clean", "spoken", "wrong"])
    assert normalized.text == "昨日学校に行きました"
    assert normalized.observed_evidence_sha256 == observed.evidence_sha256
    with pytest.raises(ValueError):
        pipeline.normalize_rank_only(observed, ["clean", "clean", "wrong"])


def test_normalization_cannot_select_outside_lattice() -> None:
    observed = MoraWeavePipeline().run(candidates()).observed
    with pytest.raises(ValueError):
        NormalizedTranscript.attach(observed, text="新しい文", mode="rank-only", selected_candidate_id="invented")


def test_rights_gate_blocks_raw_redistribution() -> None:
    registry = allow_registry()
    assert registry.require("fixture", "derive_features").asset_id == "fixture"
    with pytest.raises(PermissionError):
        registry.require("fixture", "redistribute_raw")


def test_unknown_rights_asset_fails_closed() -> None:
    with pytest.raises(PermissionError):
        allow_registry().require("unknown", "train")


def test_speaker_pseudonym_is_stable_and_does_not_contain_source_id() -> None:
    first = pseudonymize_speaker("speaker@example", b"0123456789abcdef")
    second = pseudonymize_speaker("speaker@example", b"0123456789abcdef")
    assert first == second
    assert "speaker" not in first


def test_hashed_memory_does_not_store_source_sentence() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "memory.sqlite3"
        memory = HashedNgramMemory(path)
        report = memory.ingest(["東京で音声認識を研究します"], asset_id="fixture", registry=allow_registry())
        assert report["uniqueHashedNgrams"] > 0
        assert memory.score("東京で音声認識") is not None
        memory.close()
        assert "東京で音声認識を研究します".encode("utf-8") not in path.read_bytes()


def test_teacher_probability_cache_is_digest_keyed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        cache = TeacherProbabilityCache(Path(directory) / "teacher.sqlite3")
        key = TeacherCacheKey.create(model="local", context="文脈", candidates=[{"id": "a", "text": "候補"}], audio_digest="b" * 64)
        cache.put(key, {"a": 1.0})
        assert cache.get(key) == {"a": 1.0}
        other = TeacherCacheKey.create(model="local", context="別文脈", candidates=[{"id": "a", "text": "候補"}], audio_digest="b" * 64)
        assert cache.get(other) is None


def test_teacher_probability_cache_rejects_invalid_distribution() -> None:
    with tempfile.TemporaryDirectory() as directory:
        cache = TeacherProbabilityCache(Path(directory) / "teacher.sqlite3")
        key = TeacherCacheKey.create(model="local", context="", candidates=[], audio_digest=None)
        with pytest.raises(ValueError):
            cache.put(key, {"a": 0.7, "b": 0.7})


def test_cer_and_number_error() -> None:
    assert cer("今日は3人です", "今日は2人です") == pytest.approx(1 / 6)
    assert number_error_rate("3人で1000円", "2人で1000円") == pytest.approx(0.5)


def test_missing_mora_reading_returns_null_not_false_zero() -> None:
    assert mora_error_rate(None, None) is None
    result = evaluate_transcript(reference="学校", observed="学校", normalized="学校")
    assert result.mler is None
    assert result.kana_cer is None


def test_disfluency_preservation() -> None:
    assert disfluency_preservation_rate("えっと今日は、あの、学校へ", "今日は学校へ") == 0.0
    assert disfluency_preservation_rate("えっと今日は、あの、学校へ", "えっと今日は、あの、学校へ") == 1.0


def test_unsupported_correction_rate_flags_unbacked_rewrite() -> None:
    assert unsupported_correction_rate("学校を行きました", "学校に行きました") > 0
    assert unsupported_correction_rate("学校を行きました", "学校を行きました") == 0


def test_pipeline_diagnostics_are_serializable() -> None:
    result = MoraWeavePipeline().run(candidates())
    json.dumps(result.as_dict(), ensure_ascii=False, default=str)
    assert result.diagnostics["candidateCount"] == 3
