from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from koemo.authoritative_transcription import (
    AuthoritativeTranscriptionPolicy,
    AuthoritativeTranscriptionService,
)
from koemo.semantic_asr_bridge import (
    SemanticASRBridgeResult,
    SemanticASRBridgeSegment,
)


class _Legacy:
    def __init__(self) -> None:
        self.calls = 0

    def transcribe_segments(self, audio, language="ja", on_progress=None, **kwargs):
        self.calls += 1
        assert audio == [0.1, 0.2]
        return [(0.0, 1.0, "従来の文字起こし")]


class _SemanticBridge:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0
        self.closed = False

    def transcribe_file(self, audio_path, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("semantic failure")
        return SemanticASRBridgeResult(
            observed_text="えっと料金は3000円です",
            normalized_text="料金は3,000円です。",
            decision="accepted",
            observed_evidence_sha256="a" * 64,
            segments=(
                SemanticASRBridgeSegment(
                    0,
                    1_000,
                    "えっと料金は3000円です",
                    "料金は3,000円です。",
                    "accepted",
                    "b" * 64,
                ),
            ),
            diagnostics={"selected": "candidate-a"},
            engine_version="0.2.0",
            config_sha256="c" * 64,
        )

    def close(self):
        self.closed = True


def test_semantic_asr_is_authoritative_only_when_explicitly_enabled() -> None:
    legacy = _Legacy()
    semantic = _SemanticBridge()
    service = AuthoritativeTranscriptionService(
        legacy,
        policy=AuthoritativeTranscriptionPolicy(semantic_asr_enabled=True),
        semantic_bridge=semantic,
    )
    with tempfile.TemporaryDirectory() as directory:
        audio = Path(directory) / "meeting.wav"
        audio.write_bytes(b"RIFF-fixture")
        result = service.transcribe_saved_audio(
            audio,
            legacy_audio=[0.1, 0.2],
            context="会議",
        )
    assert result.engine == "semantic-asr"
    assert result.observed_text.startswith("えっと")
    assert result.normalized_text.startswith("料金")
    assert result.decision == "accepted"
    assert semantic.calls == 1
    assert legacy.calls == 0


def test_semantic_failure_falls_back_only_when_policy_allows() -> None:
    legacy = _Legacy()
    semantic = _SemanticBridge(fail=True)
    service = AuthoritativeTranscriptionService(
        legacy,
        policy=AuthoritativeTranscriptionPolicy(
            semantic_asr_enabled=True,
            fallback_to_legacy=True,
        ),
        semantic_bridge=semantic,
    )
    with tempfile.TemporaryDirectory() as directory:
        audio = Path(directory) / "meeting.wav"
        audio.write_bytes(b"RIFF-fixture")
        result = service.transcribe_saved_audio(
            audio,
            legacy_audio=[0.1, 0.2],
        )
    assert result.engine == "legacy-faster-whisper"
    assert result.decision == "legacy-unfused"
    assert result.fallback_reason == "semantic failure"
    assert len(result.evidence_sha256) == 64
    assert legacy.calls == 1


def test_semantic_failure_is_surfaced_when_fallback_is_disabled() -> None:
    service = AuthoritativeTranscriptionService(
        _Legacy(),
        policy=AuthoritativeTranscriptionPolicy(
            semantic_asr_enabled=True,
            fallback_to_legacy=False,
        ),
        semantic_bridge=_SemanticBridge(fail=True),
    )
    with tempfile.TemporaryDirectory() as directory:
        audio = Path(directory) / "meeting.wav"
        audio.write_bytes(b"RIFF-fixture")
        with pytest.raises(RuntimeError, match="semantic failure"):
            service.transcribe_saved_audio(
                audio,
                legacy_audio=[0.1, 0.2],
            )


def test_default_policy_preserves_existing_legacy_path() -> None:
    legacy = _Legacy()
    semantic = _SemanticBridge()
    service = AuthoritativeTranscriptionService(
        legacy,
        semantic_bridge=semantic,
    )
    with tempfile.TemporaryDirectory() as directory:
        audio = Path(directory) / "meeting.wav"
        audio.write_bytes(b"RIFF-fixture")
        result = service.transcribe_saved_audio(
            audio,
            legacy_audio=[0.1, 0.2],
        )
    assert result.engine == "legacy-faster-whisper"
    assert semantic.calls == 0
    assert legacy.calls == 1
