"""Configuration adapter for Koemo's optional Semantic ASR final-pass engine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .authoritative_transcription import AuthoritativeTranscriptionPolicy
from .semantic_asr_bridge import SemanticASRBridgeConfig


@dataclass(frozen=True, slots=True)
class SemanticASRSettings:
    bridge: SemanticASRBridgeConfig
    policy: AuthoritativeTranscriptionPolicy


_BOOL_TRUE = {"1", "true", "yes", "on", "enabled"}
_BOOL_FALSE = {"0", "false", "no", "off", "disabled"}


def _bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in _BOOL_TRUE:
        return True
    if lowered in _BOOL_FALSE:
        return False
    raise ValueError(f"invalid boolean setting: {value!r}")


def _integer(value: Any, default: int, *, minimum: int = 0) -> int:
    if value is None or value == "":
        return default
    numeric = int(value)
    if numeric < minimum:
        raise ValueError(f"integer setting must be >= {minimum}")
    return numeric


def semantic_asr_settings(
    config: Mapping[str, Any],
    *,
    config_dir: str | Path | None = None,
) -> SemanticASRSettings:
    root = Path(config_dir or Path.home() / ".koemo").expanduser()
    cache_value = config.get("semantic_asr_cache")
    cache_path = (
        str(Path(str(cache_value)).expanduser())
        if cache_value
        else str(root / "semantic-asr-evidence.sqlite3")
    )
    bridge = SemanticASRBridgeConfig(
        model=str(config.get("semantic_asr_model") or config.get("model_size") or "large-v3-turbo"),
        device=str(config.get("semantic_asr_device") or "auto"),
        compute_type=str(config.get("semantic_asr_compute_type") or "default"),
        effort=str(config.get("semantic_asr_effort") or "cpu-quality"),
        cache_path=cache_path,
        language=str(config.get("language") or "ja"),
        window_ms=_integer(config.get("semantic_asr_window_ms"), 28_000, minimum=1),
        overlap_ms=_integer(config.get("semantic_asr_overlap_ms"), 1_200),
        maximum_hypotheses=_integer(
            config.get("semantic_asr_maximum_hypotheses"), 12, minimum=1
        ),
        evidence_budget_ms=_integer(
            config.get("semantic_asr_evidence_budget_ms"), 4_000
        ),
        maximum_evidence_actions=_integer(
            config.get("semantic_asr_maximum_evidence_actions"), 4
        ),
        keep_warm=_bool(config.get("semantic_asr_keep_warm"), False),
    )
    policy = AuthoritativeTranscriptionPolicy(
        semantic_asr_enabled=_bool(config.get("semantic_asr_enabled"), False),
        fallback_to_legacy=_bool(config.get("semantic_asr_fallback_to_legacy"), True),
        preserve_legacy_correction_as_normalized_only=_bool(
            config.get("semantic_asr_legacy_correction_normalized_only"), True
        ),
    )
    return SemanticASRSettings(bridge=bridge, policy=policy)
