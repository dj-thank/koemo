from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from koemo.semantic_asr_settings import semantic_asr_settings


def test_settings_are_opt_in_and_use_local_cache_by_default() -> None:
    with tempfile.TemporaryDirectory() as directory:
        settings = semantic_asr_settings({}, config_dir=directory)
    assert not settings.policy.semantic_asr_enabled
    assert settings.policy.fallback_to_legacy
    assert settings.bridge.effort == "cpu-quality"
    assert settings.bridge.cache_path == str(Path(directory) / "semantic-asr-evidence.sqlite3")


def test_settings_map_explicit_edge_profile() -> None:
    settings = semantic_asr_settings(
        {
            "semantic_asr_enabled": "true",
            "semantic_asr_fallback_to_legacy": "false",
            "semantic_asr_effort": "edge-gpu",
            "semantic_asr_model": "large-v3-turbo",
            "semantic_asr_compute_type": "float16",
            "semantic_asr_maximum_hypotheses": 16,
            "semantic_asr_evidence_budget_ms": 12000,
            "semantic_asr_keep_warm": True,
            "language": "ja",
        }
    )
    assert settings.policy.semantic_asr_enabled
    assert not settings.policy.fallback_to_legacy
    assert settings.bridge.effort == "edge-gpu"
    assert settings.bridge.maximum_hypotheses == 16
    assert settings.bridge.evidence_budget_ms == 12000
    assert settings.bridge.keep_warm


def test_settings_reject_unknown_boolean_and_effort() -> None:
    with pytest.raises(ValueError, match="invalid boolean"):
        semantic_asr_settings({"semantic_asr_enabled": "sometimes"})
    with pytest.raises(ValueError, match="effort profile"):
        semantic_asr_settings({"semantic_asr_effort": "unbounded"})


def test_settings_reject_overlap_larger_than_window() -> None:
    with pytest.raises(ValueError, match="window configuration"):
        semantic_asr_settings(
            {
                "semantic_asr_window_ms": 1000,
                "semantic_asr_overlap_ms": 1000,
            }
        )
