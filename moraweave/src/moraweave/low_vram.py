from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .acoustic_memory import AcousticCacheKey, QuerySelectedAcousticMemory
from .contracts import CandidateEvidence
from .longform import (
    LongFormConfig,
    _preservation_score,
    _safe_float,
    _segment_uncertain,
    _write_outputs,
)
from .memory import HashedNgramMemory, TeacherCacheKey, TeacherProbabilityCache
from .local_teacher import LocalTeacherClient
from .pipeline import MoraWeavePipeline


@dataclass(slots=True)
class SharedWhisperSession:
    model_name: str
    device: str = "auto"
    compute_type: str = "default"

    def __post_init__(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("install MoraWeave with the asr extra") from exc
        self.model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )

    @staticmethod
    def decode_audio(path: str | Path) -> Any:
        try:
            from faster_whisper.audio import decode_audio
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("faster-whisper audio utilities are unavailable") from exc
        return decode_audio(str(path), sampling_rate=16_000)

    def first_pass(self, waveform: Any, config: LongFormConfig) -> tuple[Any, Any]:
        return self.model.transcribe(
            waveform,
            language=config.language,
            task="transcribe",
            beam_size=config.beam_size,
            initial_prompt=config.initial_prompt,
            hotwords="、".join(config.hotwords) if config.hotwords else None,
            word_timestamps=config.word_timestamps,
            vad_filter=config.vad_filter,
            condition_on_previous_text=True,
        )

    def nbest(
        self,
        waveform: Any,
        *,
        start_ms: int,
        end_ms: int,
        language: str,
        beam_size: int,
        hypotheses: int,
    ) -> list[CandidateEvidence]:
        try:
            import numpy as np
            from faster_whisper.tokenizer import Tokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("faster-whisper N-best dependencies are unavailable") from exc
        start_sample = max(0, round(start_ms * 16))
        end_sample = min(len(waveform), round(end_ms * 16))
        if end_sample <= start_sample:
            raise ValueError("empty selective re-listening waveform")
        audio = waveform[start_sample:end_sample]
        if len(audio) / 16_000 > 30.0:
            raise ValueError("selective N-best span exceeds one Whisper window")

        tokenizer = Tokenizer(
            self.model.hf_tokenizer,
            self.model.model.is_multilingual,
            task="transcribe",
            language=language,
        )
        features = self.model.feature_extractor(audio)
        if isinstance(features, tuple):
            features = features[0]
        features = np.asarray(features)
        if features.ndim == 2:
            features = np.expand_dims(features, 0)
        encoded = self.model.encode(features)
        result = self.model.model.generate(
            encoded,
            [list(tokenizer.sot_sequence)],
            beam_size=beam_size,
            num_hypotheses=hypotheses,
            return_scores=True,
            sampling_temperature=0.0,
        )[0]

        candidates: list[CandidateEvidence] = []
        seen: set[str] = set()
        for index, (tokens, score) in enumerate(
            zip(result.sequences_ids, result.scores, strict=True)
        ):
            token_ids = tuple(int(token) for token in tokens)
            text_tokens = [token for token in token_ids if token < tokenizer.timestamp_begin]
            text = tokenizer.decode(text_tokens).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            candidates.append(
                CandidateEvidence(
                    candidate_id=f"shared-fw-{index:04d}",
                    text=text,
                    token_ids=token_ids,
                    acoustic=float(score),
                    metadata={
                        "adapter": "shared-faster-whisper-ctranslate2-nbest",
                        "model": self.model_name,
                        "startMs": start_ms,
                        "endMs": end_ms,
                    },
                )
            )
        if not candidates:
            raise RuntimeError("shared faster-whisper session returned no N-best candidate")
        return candidates


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transcribe_low_vram(
    audio_path: str | Path,
    *,
    output_dir: str | Path,
    config: LongFormConfig | None = None,
    acoustic_cache_path: str | Path | None = None,
) -> dict[str, Any]:
    """Transcribe long audio with one resident faster-whisper model.

    The source waveform is decoded once. Uncertain segments reuse the same CTranslate2
    model for local N-best decoding. This avoids the duplicate model residency of the
    generic adapter-based runtime.
    """

    config = config or LongFormConfig()
    source = Path(audio_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    source_digest = _sha256_file(source)

    session = SharedWhisperSession(config.model, config.device, config.compute_type)
    waveform = session.decode_audio(source)
    segments_iter, info = session.first_pass(waveform, config)

    memory = HashedNgramMemory(config.memory_database) if config.memory_database else None
    teacher = (
        LocalTeacherClient(model=config.teacher_model, endpoint=config.teacher_endpoint)
        if config.teacher_model
        else None
    )
    teacher_cache = TeacherProbabilityCache(config.teacher_cache) if config.teacher_cache else None
    acoustic_cache = (
        QuerySelectedAcousticMemory(acoustic_cache_path)
        if acoustic_cache_path is not None
        else None
    )
    pipeline = MoraWeavePipeline()

    observed_segments: list[dict[str, Any]] = []
    normalized_segments: list[dict[str, Any]] = []
    for index, segment in enumerate(segments_iter):
        text = str(getattr(segment, "text", "")).strip()
        start = float(getattr(segment, "start", 0.0))
        end = float(getattr(segment, "end", start))
        start_ms, end_ms = round(start * 1000), round(end * 1000)
        uncertain, uncertainty_reasons = _segment_uncertain(segment, config)
        candidates = [
            CandidateEvidence(
                candidate_id=f"seg-{index:06d}-base",
                text=text,
                acoustic=_safe_float(getattr(segment, "avg_logprob", None)),
                lexical=memory.score(text, namespace=config.memory_namespace) if memory else None,
                preservation=_preservation_score(text),
                metadata={"source": "first-pass", "uncertaintyReasons": uncertainty_reasons},
            )
        ]

        cache_hit = False
        if uncertain and end_ms > start_ms:
            cache_key = AcousticCacheKey.create(
                audio_sha256=source_digest,
                start_ms=start_ms,
                end_ms=end_ms,
                adapter="shared-faster-whisper-ctranslate2-nbest",
                model=config.model,
                beam_size=config.nbest_beam_size,
                hypotheses=config.nbest_hypotheses,
                language=config.language,
                initial_prompt=config.initial_prompt,
                hotwords=config.hotwords,
            )
            extra = acoustic_cache.get(cache_key) if acoustic_cache else None
            cache_hit = extra is not None
            if extra is None:
                extra = session.nbest(
                    waveform,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    language=config.language,
                    beam_size=config.nbest_beam_size,
                    hypotheses=config.nbest_hypotheses,
                )
                if acoustic_cache:
                    acoustic_cache.put(cache_key, extra)
            candidates.extend(extra)

        candidates = [
            dataclasses.replace(
                candidate,
                lexical=(
                    candidate.lexical
                    if candidate.lexical is not None
                    else memory.score(candidate.text, namespace=config.memory_namespace)
                    if memory
                    else None
                ),
                preservation=(
                    candidate.preservation
                    if candidate.preservation is not None
                    else _preservation_score(candidate.text)
                ),
            )
            for candidate in candidates
        ]

        probabilities: dict[str, float] | None = None
        if teacher is not None and len(candidates) > 1:
            key = TeacherCacheKey.create(
                model=config.teacher_model or "",
                context=config.initial_prompt or "",
                candidates=[{"id": item.candidate_id, "text": item.text} for item in candidates],
                audio_digest=source_digest,
            )
            probabilities = teacher_cache.get(key) if teacher_cache else None
            if probabilities is None:
                probabilities = teacher.probabilities(
                    candidates, context=config.initial_prompt or ""
                ).probabilities
                if teacher_cache:
                    teacher_cache.put(key, probabilities)
            candidates = [
                dataclasses.replace(item, teacher=probabilities.get(item.candidate_id))
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
            segment_start_ms=start_ms,
            segment_end_ms=end_ms,
            token_spans=token_spans,
        )
        diagnostics["acousticCacheHit"] = cache_hit

        normalized_candidate_id = observed.selected_candidate_id
        normalized_text = observed.text
        if probabilities:
            normalized_candidate_id = max(probabilities, key=probabilities.get)
            normalized_text = next(
                item.text for item in candidates if item.candidate_id == normalized_candidate_id
            )

        observed_segments.append(
            {
                "id": f"seg-{index:06d}",
                "start": start,
                "end": end,
                "text": observed.text,
                "selectedCandidateId": observed.selected_candidate_id,
                "evidenceSha256": observed.evidence_sha256,
                "uncertaintyReasons": uncertainty_reasons,
                "diagnostics": diagnostics,
                "candidates": [dataclasses.asdict(item) for item in candidates],
            }
        )
        normalized_segments.append(
            {
                "id": f"seg-{index:06d}",
                "start": start,
                "end": end,
                "text": normalized_text,
                "selectedCandidateId": normalized_candidate_id,
                "observedEvidenceSha256": observed.evidence_sha256,
            }
        )

    if memory:
        memory.close()
    cache_stats = acoustic_cache.stats() if acoustic_cache else None
    if acoustic_cache:
        acoustic_cache.close()

    document = {
        "schemaVersion": "1.0.0",
        "source": {"name": source.name, "bytes": source.stat().st_size, "sha256": source_digest},
        "engine": {
            "primary": "faster-whisper",
            "model": config.model,
            "device": config.device,
            "computeType": config.compute_type,
            "singleResidentModel": True,
            "sourceWaveformDecodedOnce": True,
            "selectiveNBest": True,
            "teacherModel": config.teacher_model,
            "acousticCache": cache_stats,
        },
        "language": {
            "code": getattr(info, "language", config.language),
            "probability": _safe_float(getattr(info, "language_probability", None)),
        },
        "duration": {
            "seconds": _safe_float(getattr(info, "duration", None)),
            "secondsAfterVad": _safe_float(getattr(info, "duration_after_vad", None)),
        },
        "observedTranscript": {
            "text": "".join(str(item["text"]) for item in observed_segments),
            "segments": observed_segments,
            "immutable": True,
        },
        "normalizedTranscript": {
            "text": "".join(str(item["text"]) for item in normalized_segments),
            "segments": normalized_segments,
            "derived": True,
        },
    }
    document["outputs"] = _write_outputs(document, Path(output_dir), source.stem)
    Path(document["outputs"]["json"]).write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return document
