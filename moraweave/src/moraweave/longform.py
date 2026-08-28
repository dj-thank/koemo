from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adapters import DecodeRequest, FasterWhisperAdapter, Qwen3ASRAdapter
from .contracts import CandidateEvidence
from .local_teacher import LocalTeacherClient
from .memory import HashedNgramMemory, TeacherCacheKey, TeacherProbabilityCache
from .pipeline import MoraWeavePipeline


@dataclass(slots=True)
class LongFormConfig:
    model: str = "large-v3-turbo"
    device: str = "auto"
    compute_type: str = "default"
    language: str = "ja"
    beam_size: int = 5
    nbest_beam_size: int = 12
    nbest_hypotheses: int = 8
    initial_prompt: str | None = None
    hotwords: tuple[str, ...] = ()
    vad_filter: bool = True
    word_timestamps: bool = True
    low_avg_logprob: float = -0.85
    high_no_speech_prob: float = 0.50
    low_word_probability: float = 0.52
    memory_database: str | None = None
    memory_namespace: str = "public-ja"
    teacher_model: str | None = None
    teacher_endpoint: str = "http://127.0.0.1:11434/api/chat"
    teacher_cache: str | None = None
    qwen_second_ear: bool = False


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _preservation_score(text: str) -> float:
    fillers = ("えー", "ええと", "えっと", "あの", "その", "まあ", "うーん", "んー")
    restarts = sum(text.count(marker) for marker in ("いや、", "じゃなく", "というか", "あ、"))
    filler_count = sum(text.count(filler) for filler in fillers)
    # The score does not reward errors; it rewards retaining explicit non-canonical events.
    return min(1.0, 0.55 + 0.08 * filler_count + 0.06 * restarts)


def _segment_uncertain(segment: Any, config: LongFormConfig) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    avg_logprob = _safe_float(getattr(segment, "avg_logprob", None))
    no_speech = _safe_float(getattr(segment, "no_speech_prob", None))
    if avg_logprob is not None and avg_logprob < config.low_avg_logprob:
        reasons.append("low-average-logprob")
    if no_speech is not None and no_speech > config.high_no_speech_prob:
        reasons.append("high-no-speech-probability")
    words = list(getattr(segment, "words", None) or [])
    probabilities = [
        value
        for word in words
        if (value := _safe_float(getattr(word, "probability", None))) is not None
    ]
    if probabilities and min(probabilities) < config.low_word_probability:
        reasons.append("low-word-probability")
    return bool(reasons), reasons


def _normalize_probabilities(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    maximum = max(values.values())
    exps = {key: math.exp(value - maximum) for key, value in values.items()}
    total = sum(exps.values()) or 1.0
    return {key: value / total for key, value in exps.items()}


def _subtitle_time(seconds: float, separator: str = ",") -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def _write_outputs(document: dict[str, Any], output_dir: Path, stem: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    json_path = output_dir / f"{stem}.moraweave.json"
    observed_path = output_dir / f"{stem}.observed.txt"
    normalized_path = output_dir / f"{stem}.txt"
    srt_path = output_dir / f"{stem}.srt"
    vtt_path = output_dir / f"{stem}.vtt"

    json_path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    observed_path.write_text(document["observedTranscript"]["text"].rstrip() + "\n", encoding="utf-8")
    normalized_path.write_text(document["normalizedTranscript"]["text"].rstrip() + "\n", encoding="utf-8")

    srt: list[str] = []
    vtt: list[str] = ["WEBVTT", ""]
    for index, segment in enumerate(document["normalizedTranscript"]["segments"], 1):
        start = float(segment["start"])
        end = float(segment["end"])
        text = str(segment["text"]).strip()
        srt.extend([str(index), f"{_subtitle_time(start)} --> {_subtitle_time(end)}", text, ""])
        vtt.extend([str(index), f"{_subtitle_time(start, '.')} --> {_subtitle_time(end, '.')}", text, ""])
    srt_path.write_text("\n".join(srt).rstrip() + "\n", encoding="utf-8")
    vtt_path.write_text("\n".join(vtt).rstrip() + "\n", encoding="utf-8")

    for name, path in {
        "json": json_path,
        "observedText": observed_path,
        "normalizedText": normalized_path,
        "srt": srt_path,
        "vtt": vtt_path,
    }.items():
        outputs[name] = str(path)
    return outputs


def transcribe_longform(
    audio_path: str | Path,
    *,
    output_dir: str | Path,
    config: LongFormConfig | None = None,
) -> dict[str, Any]:
    config = config or LongFormConfig()
    audio = Path(audio_path).expanduser().resolve()
    if not audio.is_file():
        raise FileNotFoundError(audio)

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("install MoraWeave with the asr extra") from exc

    source_digest = _file_sha256(audio)
    model = WhisperModel(config.model, device=config.device, compute_type=config.compute_type)
    segments_iter, info = model.transcribe(
        str(audio),
        language=config.language,
        task="transcribe",
        beam_size=config.beam_size,
        initial_prompt=config.initial_prompt,
        hotwords="、".join(config.hotwords) if config.hotwords else None,
        word_timestamps=config.word_timestamps,
        vad_filter=config.vad_filter,
        condition_on_previous_text=True,
    )

    nbest_adapter = FasterWhisperAdapter(config.model, config.device, config.compute_type)
    qwen_adapter = Qwen3ASRAdapter() if config.qwen_second_ear else None
    memory = HashedNgramMemory(config.memory_database) if config.memory_database else None
    teacher = (
        LocalTeacherClient(model=config.teacher_model, endpoint=config.teacher_endpoint)
        if config.teacher_model
        else None
    )
    teacher_cache = TeacherProbabilityCache(config.teacher_cache) if config.teacher_cache else None
    pipeline = MoraWeavePipeline()

    observed_segments: list[dict[str, Any]] = []
    normalized_segments: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(segments_iter):
        text = str(getattr(segment, "text", "")).strip()
        start = float(getattr(segment, "start", 0.0))
        end = float(getattr(segment, "end", start))
        uncertain, uncertainty_reasons = _segment_uncertain(segment, config)
        base = CandidateEvidence(
            candidate_id=f"seg-{segment_index:06d}-base",
            text=text,
            acoustic=_safe_float(getattr(segment, "avg_logprob", None)),
            lexical=memory.score(text, namespace=config.memory_namespace) if memory else None,
            preservation=_preservation_score(text),
            metadata={"source": "first-pass", "uncertaintyReasons": uncertainty_reasons},
        )
        candidates = [base]

        if uncertain:
            request = DecodeRequest(
                audio_path=str(audio),
                language=config.language,
                beam_size=config.nbest_beam_size,
                hypotheses=config.nbest_hypotheses,
                start_ms=round(start * 1000),
                end_ms=round(end * 1000),
                initial_prompt=config.initial_prompt,
                hotwords=config.hotwords,
            )
            candidates.extend(nbest_adapter.decode(request))
            if qwen_adapter is not None:
                try:
                    candidates.extend(qwen_adapter.decode(request))
                except Exception as exc:
                    base.metadata["qwenSecondEarError"] = str(exc)

        enriched: list[CandidateEvidence] = []
        for candidate in candidates:
            lexical = candidate.lexical
            if lexical is None and memory is not None:
                lexical = memory.score(candidate.text, namespace=config.memory_namespace)
            enriched.append(
                dataclasses.replace(
                    candidate,
                    lexical=lexical,
                    preservation=candidate.preservation if candidate.preservation is not None else _preservation_score(candidate.text),
                )
            )
        candidates = enriched

        teacher_probabilities: dict[str, float] | None = None
        if teacher is not None and len(candidates) > 1:
            cache_key = TeacherCacheKey.create(
                model=config.teacher_model or "",
                context=config.initial_prompt or "",
                candidates=[{"id": item.candidate_id, "text": item.text} for item in candidates],
                audio_digest=source_digest,
            )
            teacher_probabilities = teacher_cache.get(cache_key) if teacher_cache else None
            if teacher_probabilities is None:
                teacher_probabilities = teacher.probabilities(candidates, context=config.initial_prompt or "").probabilities
                if teacher_cache:
                    teacher_cache.put(cache_key, teacher_probabilities)
            candidates = [
                dataclasses.replace(item, teacher=teacher_probabilities.get(item.candidate_id))
                for item in candidates
            ]

        token_spans = [
            {
                "startMs": round(float(getattr(word, "start", start)) * 1000),
                "endMs": round(float(getattr(word, "end", end)) * 1000),
                "confidence": _safe_float(getattr(word, "probability", None)),
            }
            for word in list(getattr(segment, "words", None) or [])
        ]
        observed, _, diagnostics = pipeline.observe(
            candidates,
            source_audio_sha256=source_digest,
            segment_start_ms=round(start * 1000),
            segment_end_ms=round(end * 1000),
            token_spans=token_spans,
        )

        normalized_text = observed.text
        normalized_candidate_id = observed.selected_candidate_id
        if teacher_probabilities:
            normalized_candidate_id = max(teacher_probabilities, key=teacher_probabilities.get)
            normalized_text = next(item.text for item in candidates if item.candidate_id == normalized_candidate_id)

        observed_segments.append(
            {
                "id": f"seg-{segment_index:06d}",
                "start": start,
                "end": end,
                "text": observed.text,
                "evidenceSha256": observed.evidence_sha256,
                "selectedCandidateId": observed.selected_candidate_id,
                "uncertaintyReasons": uncertainty_reasons,
                "diagnostics": diagnostics,
                "candidates": [dataclasses.asdict(item) for item in candidates],
            }
        )
        normalized_segments.append(
            {
                "id": f"seg-{segment_index:06d}",
                "start": start,
                "end": end,
                "text": normalized_text,
                "selectedCandidateId": normalized_candidate_id,
                "observedEvidenceSha256": observed.evidence_sha256,
            }
        )

    if memory:
        memory.close()
    observed_text = "".join(segment["text"] for segment in observed_segments)
    normalized_text = "".join(segment["text"] for segment in normalized_segments)
    document = {
        "schemaVersion": "1.0.0",
        "source": {"name": audio.name, "bytes": audio.stat().st_size, "sha256": source_digest},
        "engine": {
            "primary": "faster-whisper",
            "model": config.model,
            "device": config.device,
            "computeType": config.compute_type,
            "selectiveNBest": True,
            "qwenSecondEar": config.qwen_second_ear,
            "teacherModel": config.teacher_model,
        },
        "language": {
            "code": getattr(info, "language", config.language),
            "probability": _safe_float(getattr(info, "language_probability", None)),
        },
        "duration": {
            "seconds": _safe_float(getattr(info, "duration", None)),
            "secondsAfterVad": _safe_float(getattr(info, "duration_after_vad", None)),
        },
        "observedTranscript": {"text": observed_text, "segments": observed_segments, "immutable": True},
        "normalizedTranscript": {"text": normalized_text, "segments": normalized_segments, "derived": True},
    }
    document["outputs"] = _write_outputs(document, Path(output_dir), audio.stem)
    # Re-write JSON once outputs are known.
    Path(document["outputs"]["json"]).write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return document
