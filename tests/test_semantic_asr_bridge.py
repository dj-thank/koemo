from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from koemo.semantic_asr_bridge import (
    SemanticASRBridge,
    SemanticASRBridgeConfig,
    SemanticASRUnavailable,
)


@dataclass
class _Window:
    start_ms: int
    end_ms: int
    observed_text: str
    normalized_text: str
    decision: str = "accepted"
    evidence_sha256: str = "b" * 64


@dataclass
class _Result:
    observed_text: str
    normalized_text: str
    global_evidence_sha256: str
    windows: list[_Window]
    diagnostics: dict[str, object]
    decision: str = "accepted"


class _FakeTranscriber:
    def __init__(self) -> None:
        self.calls = []

    def transcribe(self, audio_path: Path, **kwargs):
        self.calls.append((audio_path, kwargs))
        return _Result(
            observed_text="えっと料金は3000円です",
            normalized_text="料金は3,000円です。",
            global_evidence_sha256="a" * 64,
            windows=[
                _Window(
                    0,
                    2_000,
                    "えっと料金は3000円です",
                    "料金は3,000円です。",
                )
            ],
            diagnostics={"acceptedWindows": 1},
        )


class _FakeCache:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_bridge_keeps_observed_and_normalized_text_separate() -> None:
    transcriber = _FakeTranscriber()
    cache = _FakeCache()
    config = SemanticASRBridgeConfig(keep_warm=True)
    bridge = SemanticASRBridge(
        config,
        transcriber_factory=lambda _config: (transcriber, cache),
    )
    with tempfile.TemporaryDirectory() as directory:
        audio = Path(directory) / "meeting.wav"
        audio.write_bytes(b"RIFF-fixture")
        result = bridge.transcribe_file(
            audio,
            context="会議",
            hotwords=("Semantic ASR",),
        )
    assert result.observed_text == "えっと料金は3000円です"
    assert result.normalized_text == "料金は3,000円です。"
    assert result.observed_evidence_sha256 == "a" * 64
    assert result.segments[0].observed_text.startswith("えっと")
    assert result.segments[0].normalized_text.startswith("料金")
    assert result.config_sha256 == config.digest
    assert not cache.closed
    bridge.close()
    assert cache.closed


def test_bridge_unloads_optional_runtime_when_keep_warm_is_false() -> None:
    transcriber = _FakeTranscriber()
    cache = _FakeCache()
    bridge = SemanticASRBridge(
        SemanticASRBridgeConfig(keep_warm=False),
        transcriber_factory=lambda _config: (transcriber, cache),
    )
    with tempfile.TemporaryDirectory() as directory:
        audio = Path(directory) / "meeting.wav"
        audio.write_bytes(b"RIFF-fixture")
        result = bridge.transcribe_file(audio)
    assert result.decision == "accepted"
    assert cache.closed


def test_bridge_fails_explicitly_instead_of_mutating_legacy_transcript() -> None:
    def unavailable(_config):
        raise SemanticASRUnavailable("not installed")

    bridge = SemanticASRBridge(transcriber_factory=unavailable)
    with tempfile.TemporaryDirectory() as directory:
        audio = Path(directory) / "meeting.wav"
        audio.write_bytes(b"RIFF-fixture")
        with pytest.raises(SemanticASRUnavailable, match="not installed"):
            bridge.transcribe_file(audio)


def test_bridge_rejects_invalid_effort_profile() -> None:
    with pytest.raises(ValueError, match="effort profile"):
        SemanticASRBridgeConfig(effort="unbounded")
