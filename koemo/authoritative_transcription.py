"""Unified authoritative final-transcription service for Koemo.

The service prefers Semantic ASR only when explicitly enabled. It falls back to the
existing faster-whisper ``Transcriber`` only when the policy allows it and records the
fallback reason. Live preview text is never consumed here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .semantic_asr_bridge import (
    SemanticASRBridge,
    SemanticASRBridgeConfig,
    SemanticASRBridgeResult,
)


class LegacyTranscriberProtocol(Protocol):
    def transcribe_segments(
        self,
        audio: Any,
        language: str = "ja",
        on_progress: Any | None = None,
        vad_filter: bool = True,
        min_silence_duration_ms: int = 400,
        speech_pad_ms: int = 150,
    ) -> list[tuple[float, float, str]]: ...


@dataclass(frozen=True, slots=True)
class AuthoritativeTranscriptionPolicy:
    semantic_asr_enabled: bool = False
    fallback_to_legacy: bool = True
    preserve_legacy_correction_as_normalized_only: bool = True


@dataclass(frozen=True, slots=True)
class AuthoritativeSegment:
    start: float
    end: float
    text: str
    normalized_text: str
    decision: str = "accepted"
    evidence_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class AuthoritativeTranscript:
    observed_text: str
    normalized_text: str
    decision: str
    engine: str
    evidence_sha256: str
    segments: tuple[AuthoritativeSegment, ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.observed_text:
            raise ValueError("authoritative transcript must not be empty")
        if len(self.evidence_sha256) != 64:
            raise ValueError("authoritative transcript requires a SHA-256 evidence digest")


def _legacy_digest(
    audio_path: str | Path,
    segments: list[tuple[float, float, str]],
    *,
    language: str,
) -> str:
    source = Path(audio_path).expanduser().resolve()
    audio_digest = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None
    payload = json.dumps(
        {
            "engine": "legacy-faster-whisper",
            "audioSha256": audio_digest,
            "language": language,
            "segments": segments,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _semantic_result(result: SemanticASRBridgeResult) -> AuthoritativeTranscript:
    return AuthoritativeTranscript(
        observed_text=result.observed_text,
        normalized_text=result.normalized_text,
        decision=result.decision,
        engine=result.engine,
        evidence_sha256=result.observed_evidence_sha256,
        segments=tuple(
            AuthoritativeSegment(
                start=segment.start_ms / 1000.0,
                end=segment.end_ms / 1000.0,
                text=segment.observed_text,
                normalized_text=segment.normalized_text,
                decision=segment.decision,
                evidence_sha256=segment.evidence_sha256,
            )
            for segment in result.segments
        ),
        diagnostics={
            **result.diagnostics,
            "engineVersion": result.engine_version,
            "semanticAsrConfigSha256": result.config_sha256,
        },
    )


class AuthoritativeTranscriptionService:
    def __init__(
        self,
        legacy_transcriber: LegacyTranscriberProtocol,
        *,
        policy: AuthoritativeTranscriptionPolicy | None = None,
        semantic_bridge: SemanticASRBridge | None = None,
        semantic_config: SemanticASRBridgeConfig | None = None,
    ) -> None:
        self.legacy_transcriber = legacy_transcriber
        self.policy = policy or AuthoritativeTranscriptionPolicy()
        self.semantic_bridge = semantic_bridge or SemanticASRBridge(semantic_config)

    def transcribe_saved_audio(
        self,
        audio_path: str | Path,
        *,
        legacy_audio: Any | None,
        language: str = "ja",
        context: str = "",
        initial_prompt: str | None = None,
        hotwords: tuple[str, ...] = (),
        on_progress: Any | None = None,
    ) -> AuthoritativeTranscript:
        semantic_error: Exception | None = None
        if self.policy.semantic_asr_enabled:
            try:
                return _semantic_result(
                    self.semantic_bridge.transcribe_file(
                        audio_path,
                        language=language,
                        context=context,
                        initial_prompt=initial_prompt,
                        hotwords=hotwords,
                    )
                )
            except Exception as exc:
                semantic_error = exc
                if not self.policy.fallback_to_legacy:
                    raise

        if legacy_audio is None:
            reason = (
                f"Semantic ASR failed and no legacy audio buffer is available: {semantic_error}"
                if semantic_error is not None
                else "Legacy authoritative transcription requires an audio buffer"
            )
            raise RuntimeError(reason)

        segments = self.legacy_transcriber.transcribe_segments(
            legacy_audio,
            language=language,
            on_progress=on_progress,
        )
        cleaned = [
            (float(start), float(end), str(text).strip())
            for start, end, text in segments
            if str(text).strip()
        ]
        if not cleaned:
            raise RuntimeError("Legacy faster-whisper returned no authoritative text")
        text = "".join(segment[2] for segment in cleaned)
        evidence_sha256 = _legacy_digest(audio_path, cleaned, language=language)
        fallback_reason = str(semantic_error) if semantic_error is not None else None
        return AuthoritativeTranscript(
            observed_text=text,
            normalized_text=text,
            decision="legacy-unfused",
            engine="legacy-faster-whisper",
            evidence_sha256=evidence_sha256,
            segments=tuple(
                AuthoritativeSegment(
                    start=start,
                    end=end,
                    text=segment_text,
                    normalized_text=segment_text,
                    decision="legacy-unfused",
                )
                for start, end, segment_text in cleaned
            ),
            diagnostics={
                "semanticAsrEnabled": self.policy.semantic_asr_enabled,
                "fallbackToLegacy": self.policy.fallback_to_legacy,
                "legacyCorrectionBoundary": (
                    "normalized-only"
                    if self.policy.preserve_legacy_correction_as_normalized_only
                    else "legacy-existing-behavior"
                ),
            },
            fallback_reason=fallback_reason,
        )

    def close(self) -> None:
        self.semantic_bridge.close()
