from __future__ import annotations

from dataclasses import replace

import pytest

from moraweave.acquisition import EvidenceBudget, plan_evidence_acquisition
from moraweave.adapters import qwen_language_name
from moraweave.calibration import (
    ScoreRankFeatures,
    area_under_risk_coverage,
    calibrate_values,
    expected_calibration_error,
    score_rank_confidence,
)
from moraweave.contracts import CandidateEvidence, ObservedTranscript, canonical_json
from moraweave.evaluation import critical_entity_error_rate, evaluate_confidence, oracle_cer
from moraweave.gates import GateConfig, gate_candidates
from moraweave.local_teacher import OpenAICompatibleTeacherClient, validate_openai_endpoint
from moraweave.pipeline import MoraWeavePipeline
from moraweave.shadow_lattice import build_dual_evidence_lattice


def test_canonical_json_recurses_into_nested_dataclasses() -> None:
    ranked = gate_candidates(
        [
            CandidateEvidence("a", "東京", acoustic=0.9, mora=0.8),
            CandidateEvidence("b", "京都", acoustic=0.1, mora=0.2),
        ]
    )
    observed = ObservedTranscript.create(selected=ranked[0], ranked=ranked, uncertainty_spans=[])
    observed.verify()
    assert "candidate_id" in canonical_json(observed)


def test_probability_evidence_is_not_minmax_distorted() -> None:
    assert calibrate_values([0.51, 0.50, 0.0]) == [0.51, 0.50, 0.0]


def test_raw_score_fallback_is_robust_to_outlier() -> None:
    calibrated = calibrate_values([-0.1, -0.2, -20.0])
    assert calibrated[0] > calibrated[1] > calibrated[2]
    assert calibrated[1] > 0.4


def test_score_rank_confidence_uses_rank_and_margin() -> None:
    top = score_rank_confidence(
        ScoreRankFeatures(rank=1, hypothesis_count=5, avg_logprob=-0.1, margin_to_next=0.3)
    )
    lower = score_rank_confidence(
        ScoreRankFeatures(rank=4, hypothesis_count=5, avg_logprob=-1.2, margin_to_next=0.01)
    )
    assert top > lower


def test_missing_evidence_becomes_provisional() -> None:
    config = replace(
        GateConfig.default(),
        minimum_evidence_coverage=0.8,
        acceptance_posterior=0.8,
    )
    candidates = [
        CandidateEvidence("a", "東京です", acoustic=0.52),
        CandidateEvidence("b", "東京でした", acoustic=0.50),
    ]
    ranked = gate_candidates(candidates, config)
    assert ranked[0].gate.needs_relisten
    assert ranked[0].gate.abstain
    assert MoraWeavePipeline(config).run(candidates).observed.decision == "provisional"


def test_mora_shadow_collapses_surface_only_difference() -> None:
    lattice = build_dual_evidence_lattice(
        [
            CandidateEvidence("kanji", "今日", acoustic=0.8, reading="キョウ"),
            CandidateEvidence("kana", "きょう", acoustic=0.7, reading="キョウ"),
        ]
    )
    assert lattice.alignment_level == "mora"
    assert lattice.contradiction_islands == ()


def test_number_island_receives_high_criticality() -> None:
    candidates = [
        CandidateEvidence("a", "三人です", acoustic=0.6),
        CandidateEvidence("b", "二人です", acoustic=0.5),
    ]
    lattice = build_dual_evidence_lattice(candidates, posterior={"a": 0.55, "b": 0.45})
    island = lattice.contradiction_islands[0]
    assert "number" in island.kinds
    assert island.criticality == 1.0
    assert island.posterior_ambiguity > 0.9


def test_budgeted_acquisition_prefers_useful_actions() -> None:
    candidates = [
        CandidateEvidence("a", "三人です", acoustic=0.51, mora=0.5),
        CandidateEvidence("b", "二人です", acoustic=0.50, mora=0.51),
    ]
    ranked = gate_candidates(candidates)
    lattice = build_dual_evidence_lattice(candidates, posterior=ranked[0].gate.posterior)
    plan = plan_evidence_acquisition(
        ranked,
        lattice,
        budget=EvidenceBudget(total_cost_ms=2500, max_actions=2),
    )
    assert plan.selected
    assert plan.used_ms <= 2500
    assert len(plan.selected) <= 2
    assert all(action.utility > 0 for action in plan.selected)


def test_qwen_language_mapping_matches_official_api() -> None:
    assert qwen_language_name("ja") == "Japanese"
    assert qwen_language_name("日本語") == "Japanese"
    assert qwen_language_name(None) is None


def test_qwen38_openai_teacher_is_loopback_only() -> None:
    endpoint = validate_openai_endpoint("http://127.0.0.1:8000")
    client = OpenAICompatibleTeacherClient(endpoint=endpoint)
    assert endpoint.endswith("/v1/chat/completions")
    assert client.preserve_thinking is False
    with pytest.raises(ValueError):
        validate_openai_endpoint("http://example.com/v1/chat/completions")


def test_calibration_and_selective_metrics() -> None:
    confidence = [0.95, 0.8, 0.55, 0.2]
    correct = [1, 1, 0, 0]
    metrics = evaluate_confidence(confidence, correct)
    assert 0 <= metrics.expected_calibration_error <= 1
    assert 0 <= metrics.brier <= 1
    assert metrics.aurc == pytest.approx(area_under_risk_coverage(confidence, correct))
    assert expected_calibration_error(confidence, correct) < 0.3


def test_oracle_and_critical_entity_metrics() -> None:
    assert oracle_cer("今日は三人です", ["今日は二人です", "今日は三人です"]) == 0
    assert critical_entity_error_rate("Qwen3.8を三人で使う", "Qwen3.7を二人で使う") > 0
