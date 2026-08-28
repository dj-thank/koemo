from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .types import EngineResult, Segment, Word


@dataclass(slots=True)
class EngineConfig:
    model: str = "large-v3-turbo"
    device: str = "auto"
    compute_type: str = "default"
    language: str | None = "ja"
    task: str = "transcribe"
    beam_size: int = 5
    best_of: int = 5
    temperature: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    vad_filter: bool = True
    vad_threshold: float = 0.5
    min_silence_duration_ms: int = 500
    speech_pad_ms: int = 300
    max_speech_duration_s: float = 30.0
    word_timestamps: bool = True
    condition_on_previous_text: bool = True
    initial_prompt: str | None = None
    hotwords: str | None = None
    log_progress: bool = False


class FasterWhisperEngine:
    def __init__(self, config: EngineConfig) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("faster-whisper is required; install the project with the 'asr' extra") from exc
        self.config = config
        self._model = WhisperModel(config.model, device=config.device, compute_type=config.compute_type)

    @staticmethod
    def _package_version(name: str) -> str | None:
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    def transcribe(self, audio_path: str | Path) -> EngineResult:
        config = self.config
        language = None if config.language in {None, "", "auto"} else config.language
        vad_parameters = None
        if config.vad_filter:
            vad_parameters = {
                "threshold": config.vad_threshold,
                "min_silence_duration_ms": config.min_silence_duration_ms,
                "speech_pad_ms": config.speech_pad_ms,
                "max_speech_duration_s": config.max_speech_duration_s,
            }

        segments_iter, info = self._model.transcribe(
            str(audio_path),
            language=language,
            task=config.task,
            beam_size=config.beam_size,
            best_of=config.best_of,
            temperature=list(config.temperature),
            condition_on_previous_text=config.condition_on_previous_text,
            initial_prompt=config.initial_prompt,
            hotwords=config.hotwords,
            word_timestamps=config.word_timestamps,
            vad_filter=config.vad_filter,
            vad_parameters=vad_parameters,
            log_progress=config.log_progress,
        )

        segments: list[Segment] = []
        word_index = 0
        for index, raw_segment in enumerate(segments_iter):
            words: list[Word] = []
            for raw_word in list(getattr(raw_segment, "words", None) or []):
                words.append(
                    Word(
                        index=word_index,
                        text=str(getattr(raw_word, "word", "")),
                        start=_float_or_none(getattr(raw_word, "start", None)),
                        end=_float_or_none(getattr(raw_word, "end", None)),
                        probability=_float_or_none(getattr(raw_word, "probability", None)),
                    )
                )
                word_index += 1

            raw_id = getattr(raw_segment, "id", index)
            try:
                numeric_id = int(raw_id)
            except (TypeError, ValueError):
                numeric_id = index
            segments.append(
                Segment(
                    id=f"seg-{numeric_id:06d}",
                    index=index,
                    start=float(getattr(raw_segment, "start", 0.0)),
                    end=float(getattr(raw_segment, "end", 0.0)),
                    text=str(getattr(raw_segment, "text", "")),
                    seek=_int_or_none(getattr(raw_segment, "seek", None)),
                    temperature=_float_or_none(getattr(raw_segment, "temperature", None)),
                    avg_logprob=_float_or_none(getattr(raw_segment, "avg_logprob", None)),
                    compression_ratio=_float_or_none(getattr(raw_segment, "compression_ratio", None)),
                    no_speech_prob=_float_or_none(getattr(raw_segment, "no_speech_prob", None)),
                    words=words,
                )
            )

        return EngineResult(
            engine={
                "name": "faster-whisper",
                "version": self._package_version("faster-whisper"),
                "ctranslate2Version": self._package_version("ctranslate2"),
                "model": config.model,
                "device": config.device,
                "computeType": config.compute_type,
                "task": config.task,
                "beamSize": config.beam_size,
                "bestOf": config.best_of,
                "temperature": list(config.temperature),
                "vadFilter": config.vad_filter,
                "wordTimestamps": config.word_timestamps,
                "conditionOnPreviousText": config.condition_on_previous_text,
                "initialPromptUsed": bool(config.initial_prompt),
                "hotwordsUsed": bool(config.hotwords),
            },
            language={
                "code": getattr(info, "language", language),
                "probability": _float_or_none(getattr(info, "language_probability", None)),
                "allProbabilities": _jsonable(getattr(info, "all_language_probs", None)),
            },
            duration={
                "seconds": _float_or_none(getattr(info, "duration", None)),
                "secondsAfterVad": _float_or_none(getattr(info, "duration_after_vad", None)),
            },
            segments=segments,
        )


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)
