from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .contracts import CandidateEvidence


@dataclass(frozen=True, slots=True)
class DecodeRequest:
    audio_path: str
    language: str = "ja"
    beam_size: int = 5
    hypotheses: int = 5
    start_ms: int | None = None
    end_ms: int | None = None
    initial_prompt: str | None = None
    hotwords: tuple[str, ...] = ()


class ASRAdapter(Protocol):
    name: str

    def decode(self, request: DecodeRequest) -> list[CandidateEvidence]: ...


class MockASRAdapter:
    name = "mock"

    def __init__(self, candidates: list[CandidateEvidence]) -> None:
        self.candidates = candidates
        self.requests: list[DecodeRequest] = []

    def decode(self, request: DecodeRequest) -> list[CandidateEvidence]:
        self.requests.append(request)
        return list(self.candidates[: request.hypotheses])


class FasterWhisperAdapter:
    """Optional N-best adapter using faster-whisper's wrapped CTranslate2 model.

    This adapter intentionally limits each call to one Whisper window. Long audio must be
    segmented first; selective re-listening calls it on local spans.
    """

    name = "faster-whisper-ctranslate2-nbest"

    def __init__(self, model: str = "large-v3-turbo", device: str = "auto", compute_type: str = "default") -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("install MoraWeave with the asr extra") from exc
        self.model_name = model
        self.model = WhisperModel(model, device=device, compute_type=compute_type)

    def decode(self, request: DecodeRequest) -> list[CandidateEvidence]:
        try:
            from faster_whisper.audio import decode_audio
            from faster_whisper.tokenizer import Tokenizer
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("faster-whisper runtime dependencies are unavailable") from exc

        waveform = decode_audio(request.audio_path, sampling_rate=16_000)
        if request.start_ms is not None or request.end_ms is not None:
            start = max(0, int((request.start_ms or 0) * 16))
            end = min(len(waveform), int((request.end_ms or len(waveform) / 16) * 16))
            waveform = waveform[start:end]
        if len(waveform) / 16_000 > 30.0:
            raise ValueError("decode request exceeds one Whisper window")

        tokenizer = Tokenizer(
            self.model.hf_tokenizer,
            self.model.model.is_multilingual,
            task="transcribe",
            language=request.language,
        )
        features = self.model.feature_extractor(waveform)
        if isinstance(features, tuple):
            features = features[0]
        features = np.asarray(features)
        if features.ndim == 2:
            features = np.expand_dims(features, 0)
        encoded = self.model.encode(features)
        result = self.model.model.generate(
            encoded,
            [list(tokenizer.sot_sequence)],
            beam_size=request.beam_size,
            num_hypotheses=request.hypotheses,
            return_scores=True,
            sampling_temperature=0.0,
        )[0]

        output: list[CandidateEvidence] = []
        for index, (tokens, score) in enumerate(zip(result.sequences_ids, result.scores, strict=True)):
            token_ids = tuple(int(token) for token in tokens)
            text_tokens = [token for token in token_ids if token < tokenizer.timestamp_begin]
            text = tokenizer.decode(text_tokens).strip()
            if text:
                output.append(
                    CandidateEvidence(
                        candidate_id=f"fw-{index:04d}",
                        text=text,
                        token_ids=token_ids,
                        acoustic=float(score),
                        metadata={"adapter": self.name, "model": self.model_name},
                    )
                )
        if not output:
            raise RuntimeError("faster-whisper returned no non-empty hypothesis")
        return output


class Qwen3ASRAdapter:
    """Optional second-ear adapter using the official qwen-asr package API.

    Because upstream APIs can change, this class keeps the import and response adaptation
    isolated. Real inference is covered by a target-machine compatibility test, not core CI.
    """

    name = "qwen3-asr"

    def __init__(self, model: str = "Qwen/Qwen3-ASR-0.6B") -> None:
        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("install MoraWeave with the qwen extra") from exc
        self.model_name = model
        self.model = Qwen3ASRModel.from_pretrained(model)

    def decode(self, request: DecodeRequest) -> list[CandidateEvidence]:
        result: Any = self.model.transcribe(
            audio=request.audio_path,
            language=request.language,
            context=request.initial_prompt,
        )
        rows = result if isinstance(result, list) else [result]
        candidates: list[CandidateEvidence] = []
        for index, row in enumerate(rows[: request.hypotheses]):
            text = getattr(row, "text", None) or (row.get("text") if isinstance(row, dict) else None)
            if text:
                candidates.append(
                    CandidateEvidence(
                        candidate_id=f"qwen-{index:04d}",
                        text=str(text).strip(),
                        metadata={"adapter": self.name, "model": self.model_name},
                    )
                )
        if not candidates:
            raise RuntimeError("Qwen3-ASR returned no transcript")
        return candidates
