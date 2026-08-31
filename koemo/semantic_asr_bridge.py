"""Optional bridge from Koemo recordings to the Semantic ASR evidence core.

This module deliberately does not import Semantic ASR at Koemo startup. The dependency
is lazy, optional, and used only for the authoritative post-recording pass. Windows
Speech and rolling Whisper remain preview engines.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol


class SemanticASRUnavailable(RuntimeError):
    """Raised when the optional Semantic ASR runtime cannot be constructed."""


@dataclass(frozen=True, slots=True)
class SemanticASRBridgeConfig:
    model: str = "large-v3-turbo"
    device: str = "auto"
    compute_type: str = "default"
    effort: str = "cpu-quality"
    cache_path: str | None = None
    language: str = "ja"
    window_ms: int = 28_000
    overlap_ms: int = 1_200
    maximum_hypotheses: int = 12
    evidence_budget_ms: int = 4_000
    maximum_evidence_actions: int = 4
    keep_warm: bool = False

    def __post_init__(self) -> None:
        if self.effort not in {"ultra-light", "cpu-quality", "edge-gpu", "research"}:
            raise ValueError("unknown Semantic ASR effort profile")
        if self.window_ms <= 0 or not 0 <= self.overlap_ms < self.window_ms:
            raise ValueError("invalid Semantic ASR window configuration")
        if self.maximum_hypotheses < 1:
            raise ValueError("maximum_hypotheses must be positive")
        if self.evidence_budget_ms < 0 or self.maximum_evidence_actions < 0:
            raise ValueError("evidence budgets must be non-negative")

    @property
    def digest(self) -> str:
        payload = json.dumps(
            dataclasses.asdict(self),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class SemanticASRBridgeSegment:
    start_ms: int
    end_ms: int
    observed_text: str
    normalized_text: str
    decision: str
    evidence_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.start_ms < 0 or self.end_ms < self.start_ms:
            raise ValueError("invalid Semantic ASR bridge segment timing")
        if not self.observed_text:
            raise ValueError("bridge segment observed text must not be empty")


@dataclass(frozen=True, slots=True)
class SemanticASRBridgeResult:
    observed_text: str
    normalized_text: str
    decision: str
    observed_evidence_sha256: str
    segments: tuple[SemanticASRBridgeSegment, ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    engine: str = "semantic-asr"
    engine_version: str | None = None
    config_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.observed_text:
            raise ValueError("authoritative observed transcript must not be empty")
        if len(self.observed_evidence_sha256) != 64:
            raise ValueError("observed evidence digest must be SHA-256 hex")


class _TranscriberProtocol(Protocol):
    def transcribe(self, audio_path: Path, **kwargs: Any) -> Any: ...


TranscriberFactory = Callable[[SemanticASRBridgeConfig], tuple[_TranscriberProtocol, Any | None]]


def _package_version() -> str | None:
    try:
        return importlib.metadata.version("semantic-asr")
    except importlib.metadata.PackageNotFoundError:
        return None


def _default_factory(
    config: SemanticASRBridgeConfig,
) -> tuple[_TranscriberProtocol, Any | None]:
    try:
        from semantic_asr.advanced_adapters import PathPreservingFasterWhisperAdapter
        from semantic_asr.cache import EvidenceCache
        from semantic_asr.longform import SemanticASRTranscriber
        from semantic_asr.planner import EvidenceBudget
    except Exception as exc:  # pragma: no cover - optional runtime
        raise SemanticASRUnavailable(
            "Semantic ASR is not installed. Install the pinned semantic-asr[asr] "
            "revision or keep Koemo's legacy final transcription path."
        ) from exc

    try:
        adapter = PathPreservingFasterWhisperAdapter(
            model=config.model,
            device=config.device,
            compute_type=config.compute_type,
        )
        cache = EvidenceCache(config.cache_path) if config.cache_path else None
        transcriber = SemanticASRTranscriber(
            adapter,
            cache=cache,
            evidence_budget=EvidenceBudget(
                total_cost_ms=config.evidence_budget_ms,
                max_actions=config.maximum_evidence_actions,
            ),
            balanced_router=True,
            window_ms=config.window_ms,
            overlap_ms=config.overlap_ms,
        )
        return transcriber, cache
    except Exception as exc:  # pragma: no cover - model/runtime dependent
        raise SemanticASRUnavailable(f"Semantic ASR initialization failed: {exc}") from exc


def _plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {key: _plain(item) for key, item in dataclasses.asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _first(payload: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in payload and payload[name] is not None:
            return payload[name]
    return default


def _evidence_digest(payload: Mapping[str, Any]) -> str:
    candidate = _first(
        payload,
        "global_evidence_sha256",
        "globalEvidenceSha256",
        "evidence_sha256",
        "evidenceSha256",
        "observed_evidence_sha256",
        "observedEvidenceSha256",
    )
    if isinstance(candidate, str) and len(candidate) == 64:
        return candidate
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _segments(payload: Mapping[str, Any]) -> tuple[SemanticASRBridgeSegment, ...]:
    raw_windows = _first(payload, "windows", "segments", default=[])
    output: list[SemanticASRBridgeSegment] = []
    if not isinstance(raw_windows, list):
        return ()
    for raw in raw_windows:
        if not isinstance(raw, Mapping):
            continue
        observed = str(
            _first(
                raw,
                "observed_text",
                "observedText",
                "text",
                default="",
            )
            or ""
        ).strip()
        if not observed:
            nested = raw.get("observed")
            if isinstance(nested, Mapping):
                observed = str(_first(nested, "text", default="") or "").strip()
        if not observed:
            continue
        normalized = str(
            _first(raw, "normalized_text", "normalizedText", default=observed)
            or observed
        ).strip()
        decision = str(_first(raw, "decision", default="accepted") or "accepted")
        output.append(
            SemanticASRBridgeSegment(
                start_ms=int(_first(raw, "start_ms", "startMs", default=0) or 0),
                end_ms=int(_first(raw, "end_ms", "endMs", default=0) or 0),
                observed_text=observed,
                normalized_text=normalized or observed,
                decision=decision,
                evidence_sha256=(
                    str(_first(raw, "evidence_sha256", "evidenceSha256"))
                    if _first(raw, "evidence_sha256", "evidenceSha256")
                    else None
                ),
            )
        )
    return tuple(output)


def _bridge_result(raw_result: Any, config: SemanticASRBridgeConfig) -> SemanticASRBridgeResult:
    payload = _plain(raw_result)
    if not isinstance(payload, Mapping):
        raise RuntimeError("Semantic ASR returned an unsupported result object")
    observed = str(
        _first(payload, "observed_text", "observedText", default="") or ""
    ).strip()
    normalized = str(
        _first(payload, "normalized_text", "normalizedText", default=observed)
        or observed
    ).strip()
    segments = _segments(payload)
    if not observed and segments:
        observed = "".join(segment.observed_text for segment in segments)
    if not normalized and segments:
        normalized = "".join(segment.normalized_text for segment in segments)
    if not observed:
        raise RuntimeError("Semantic ASR returned no authoritative observed text")
    diagnostics = _first(payload, "diagnostics", default={})
    if not isinstance(diagnostics, Mapping):
        diagnostics = {"rawDiagnostics": diagnostics}
    decision = str(
        _first(payload, "decision", "observation_decision", default="accepted")
        or "accepted"
    )
    return SemanticASRBridgeResult(
        observed_text=observed,
        normalized_text=normalized or observed,
        decision=decision,
        observed_evidence_sha256=_evidence_digest(payload),
        segments=segments,
        diagnostics=dict(diagnostics),
        engine_version=_package_version(),
        config_sha256=config.digest,
    )


class SemanticASRBridge:
    """Thread-safe lazy authoritative final-pass bridge for saved Koemo audio."""

    def __init__(
        self,
        config: SemanticASRBridgeConfig | None = None,
        *,
        transcriber_factory: TranscriberFactory | None = None,
    ) -> None:
        self.config = config or SemanticASRBridgeConfig()
        self._factory = transcriber_factory or _default_factory
        self._transcriber: _TranscriberProtocol | None = None
        self._cache: Any | None = None
        self._lock = threading.RLock()

    @staticmethod
    def dependency_available() -> bool:
        try:
            return _package_version() is not None
        except Exception:
            return False

    def _ensure_runtime(self) -> _TranscriberProtocol:
        if self._transcriber is None:
            self._transcriber, self._cache = self._factory(self.config)
        return self._transcriber

    def transcribe_file(
        self,
        audio_path: str | Path,
        *,
        language: str | None = None,
        context: str = "",
        initial_prompt: str | None = None,
        hotwords: tuple[str, ...] = (),
        duration_ms: int | None = None,
    ) -> SemanticASRBridgeResult:
        source = Path(audio_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        with self._lock:
            transcriber = self._ensure_runtime()
            try:
                raw = transcriber.transcribe(
                    source,
                    duration_ms=duration_ms,
                    language=language or self.config.language,
                    initial_prompt=initial_prompt,
                    hotwords=hotwords,
                    context=context,
                )
            except Exception as exc:
                raise RuntimeError(f"Semantic ASR final transcription failed: {exc}") from exc
            result = _bridge_result(raw, self.config)
            if not self.config.keep_warm:
                self.close()
            return result

    def close(self) -> None:
        with self._lock:
            cache = self._cache
            self._transcriber = None
            self._cache = None
            if cache is not None and hasattr(cache, "close"):
                cache.close()
