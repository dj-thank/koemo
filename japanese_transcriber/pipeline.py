from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from .diarization import assign_speakers, parse_rttm, relabel_speakers
from .formatters import write_outputs
from .local_llm import OllamaNormalizer
from .normalization import deterministic_normalize, join_fragments
from .quality import QualityThresholds, annotate_segment, summarize_quality
from .reading import reading_and_mora
from .types import EngineResult, Segment


class TranscriptionEngine(Protocol):
    def transcribe(self, audio_path: str | Path) -> EngineResult: ...


@dataclass(slots=True)
class PipelineConfig:
    output_dir: Path
    formats: set[str] = field(default_factory=lambda: {"json", "txt", "observed-txt", "md", "srt", "vtt", "tsv", "words-jsonl"})
    overwrite: bool = False
    rttm_path: Path | None = None
    annotate_mora: bool = True
    use_pyopenjtalk: bool = False
    quality: QualityThresholds = field(default_factory=QualityThresholds)
    local_normalizer: OllamaNormalizer | None = None
    normalization_context: str = ""
    include_source_path: bool = False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _word_dict(word) -> dict[str, object]:
    return {
        "index": word.index,
        "text": word.text,
        "start": word.start,
        "end": word.end,
        "probability": word.probability,
        "speaker": word.speaker,
        "reading": word.reading,
        "mora": list(word.mora),
    }


def _segment_dict(segment: Segment, *, text: str | None = None) -> dict[str, object]:
    return {
        "id": segment.id,
        "index": segment.index,
        "start": segment.start,
        "end": segment.end,
        "speaker": segment.speaker,
        "text": segment.text if text is None else text,
        "seek": segment.seek,
        "temperature": segment.temperature,
        "avgLogprob": segment.avg_logprob,
        "compressionRatio": segment.compression_ratio,
        "noSpeechProb": segment.no_speech_prob,
        "uncertaintyReasons": list(segment.uncertainty_reasons),
        "words": [_word_dict(word) for word in segment.words],
    }


def _annotate_mora(segments: list[Segment], *, use_pyopenjtalk: bool) -> None:
    for segment in segments:
        for word in segment.words:
            reading, mora = reading_and_mora(word.text, use_pyopenjtalk=use_pyopenjtalk)
            word.reading = reading
            word.mora = mora


def _normalized_segments(segments: list[Segment], local_normalizer: OllamaNormalizer | None, context: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    deterministic = {segment.id: deterministic_normalize(segment.text) for segment in segments}
    metadata: dict[str, object] = {"mode": "deterministic", "model": None, "rejectedSegmentIds": []}

    if local_normalizer is not None:
        result = local_normalizer.normalize(segments, context=context)
        deterministic.update({row["id"]: row["text"] for row in result.segments})
        metadata = {
            "mode": "local-llm-guarded",
            "model": result.model,
            "endpointOrigin": result.endpoint_origin,
            "rejectedSegmentIds": result.rejected_segment_ids,
        }

    return ([_segment_dict(segment, text=deterministic[segment.id]) for segment in segments], metadata)


def verify_observed_integrity(document: dict[str, object]) -> bool:
    observed = document.get("observedTranscript")
    if not isinstance(observed, dict):
        raise ValueError("observedTranscript is missing")
    expected = observed.get("sha256")
    payload = {"text": observed.get("text"), "segments": observed.get("segments")}
    actual = _sha256_json(payload)
    if expected != actual:
        raise ValueError("observed transcript evidence was modified")
    normalized = document.get("normalizedTranscript")
    if isinstance(normalized, dict) and normalized.get("observedSha256") != expected:
        raise ValueError("normalized transcript is detached from observed evidence")
    return True


def transcribe_file(audio_path: str | Path, *, engine: TranscriptionEngine, config: PipelineConfig) -> tuple[dict[str, object], dict[str, str]]:
    source = Path(audio_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    result = engine.transcribe(source)
    segments = result.segments
    for segment in segments:
        annotate_segment(segment, config.quality)

    speaker_mapping: dict[str, str] = {}
    if config.rttm_path is not None:
        turns = parse_rttm(config.rttm_path)
        assign_speakers(segments, turns)
        speaker_mapping = relabel_speakers(segments)

    if config.annotate_mora:
        _annotate_mora(segments, use_pyopenjtalk=config.use_pyopenjtalk)

    observed_segments = [_segment_dict(segment) for segment in segments]
    observed_text = join_fragments(segment.text for segment in segments)
    observed_payload = {"text": observed_text, "segments": observed_segments}
    observed_sha256 = _sha256_json(observed_payload)

    normalized_segments, normalization_metadata = _normalized_segments(segments, config.local_normalizer, config.normalization_context)
    normalized_text = join_fragments(str(segment["text"]) for segment in normalized_segments)

    document: dict[str, object] = {
        "schemaVersion": "1.0.0",
        "createdAt": datetime.now(UTC).isoformat(),
        "source": {
            "name": source.name,
            "path": str(source) if config.include_source_path else source.name,
            "bytes": source.stat().st_size,
            "sha256": _sha256_file(source),
        },
        "engine": result.engine,
        "language": result.language,
        "duration": result.duration,
        "observedTranscript": {**observed_payload, "sha256": observed_sha256, "immutable": True},
        "normalizedTranscript": {
            "text": normalized_text,
            "segments": normalized_segments,
            "observedSha256": observed_sha256,
            **normalization_metadata,
        },
        "diarization": {
            "enabled": config.rttm_path is not None,
            "source": str(config.rttm_path) if config.rttm_path else None,
            "speakerMapping": speaker_mapping,
        },
        "quality": summarize_quality(segments),
    }

    verify_observed_integrity(document)
    outputs = write_outputs(document, config.output_dir, stem=source.stem, formats=config.formats, overwrite=config.overwrite, variant="normalized")
    document["outputs"] = outputs

    if "json" in outputs:
        from .formatters import atomic_write
        atomic_write(outputs["json"], json.dumps(document, ensure_ascii=False, indent=2) + "\n", overwrite=True)

    return document, outputs
