from __future__ import annotations

import hashlib
import json
import subprocess
import wave
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Protocol

from .adapters import ASRAdapter, DecodeRequest
from .contracts import CandidateEvidence, NormalizedTranscript, ObservedTranscript, sha256_json
from .gates import GateConfig, evidence_summary, gate_candidates
from .local_teacher import DelayedTeacherPolicy, TeacherResult
from .pipeline import MoraWeavePipeline
from .runtime_cache import RuntimeCacheKey, RuntimeEvidenceCache, TeacherCacheEntry
from .selective import merge_relisten_candidates, plan_relisten


class TeacherClient(Protocol):
    model: str

    def probabilities(
        self,
        candidates: list[CandidateEvidence],
        *,
        context: str = "",
        locked_consensus: str = "",
        contradiction: str = "",
    ) -> TeacherResult: ...


@dataclass(frozen=True, slots=True)
class Window:
    index: int
    start_ms: int
    end_ms: int


@dataclass(frozen=True, slots=True)
class LongformSegment:
    window: Window
    observed: ObservedTranscript
    normalized: NormalizedTranscript
    diagnostics: dict[str, Any]
    cache_hits: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LongformResult:
    source_path: str
    source_audio_sha256: str
    duration_ms: int
    observed_text: str
    normalized_text: str
    segments: tuple[LongformSegment, ...]
    evidence_sha256: str
    diagnostics: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_duration_ms(path: str | Path) -> int:
    source = Path(path)
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(source),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        seconds = float(completed.stdout.strip())
        if seconds > 0:
            return max(1, round(seconds * 1000))
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        pass
    if source.suffix.lower() == ".wav":
        with wave.open(str(source), "rb") as audio:
            return max(1, round(audio.getnframes() / audio.getframerate() * 1000))
    raise RuntimeError("ffprobe is required to determine non-WAV duration")


def plan_windows(
    duration_ms: int,
    *,
    window_ms: int = 28_000,
    overlap_ms: int = 1_200,
) -> list[Window]:
    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive")
    if window_ms <= 0 or overlap_ms < 0 or overlap_ms >= window_ms:
        raise ValueError("window and overlap configuration is invalid")
    windows: list[Window] = []
    cursor = 0
    index = 0
    while cursor < duration_ms:
        end = min(duration_ms, cursor + window_ms)
        windows.append(Window(index=index, start_ms=cursor, end_ms=end))
        if end >= duration_ms:
            break
        cursor = end - overlap_ms
        index += 1
    return windows


def _ascii_boundary(left: str, right: str) -> bool:
    return bool(left and right and left[-1].isascii() and right[0].isascii() and left[-1].isalnum() and right[0].isalnum())


def stitch_text(left: str, right: str, *, maximum_overlap_chars: int = 160) -> str:
    left = str(left or "").strip()
    right = str(right or "").strip()
    if not left:
        return right
    if not right:
        return left
    maximum = min(maximum_overlap_chars, len(left), len(right))
    overlap = 0
    for length in range(maximum, 1, -1):
        if left[-length:] == right[:length]:
            overlap = length
            break
    suffix = right[overlap:]
    if not suffix:
        return left
    separator = " " if _ascii_boundary(left, suffix) else ""
    if left[-1:] in "、。！？!?" and suffix[:1] == left[-1:]:
        suffix = suffix[1:]
    return left + separator + suffix


def _adapter_model(adapter: ASRAdapter) -> str:
    return str(getattr(adapter, "model_name", getattr(adapter, "name", type(adapter).__name__)))


def _scoped_candidates(
    candidates: Iterable[CandidateEvidence],
    *,
    namespace: str,
    start_ms: int,
    end_ms: int,
) -> list[CandidateEvidence]:
    output: list[CandidateEvidence] = []
    for index, candidate in enumerate(candidates, 1):
        metadata = dict(candidate.metadata)
        metadata.setdefault("sourceSupport", [candidate.evidence_source])
        metadata["decodeNamespace"] = namespace
        metadata["decodeStartMs"] = start_ms
        metadata["decodeEndMs"] = end_ms
        output.append(
            replace(
                candidate,
                candidate_id=f"{namespace}:{start_ms}-{end_ms}:{index:04d}",
                source=candidate.evidence_source,
                metadata=metadata,
            )
        )
    return output


class FrontierLongformTranscriber:
    """Long-form runtime with cached, uncertainty-triggered evidence acquisition."""

    def __init__(
        self,
        base_adapter: ASRAdapter,
        *,
        second_ear: ASRAdapter | None = None,
        teacher: TeacherClient | None = None,
        cache: RuntimeEvidenceCache | None = None,
        gate_config: GateConfig | None = None,
        teacher_policy: DelayedTeacherPolicy | None = None,
        window_ms: int = 28_000,
        overlap_ms: int = 1_200,
        relisten_budget_ms: int = 10_000,
    ) -> None:
        self.base_adapter = base_adapter
        self.second_ear = second_ear
        self.teacher = teacher
        self.cache = cache
        self.gate_config = gate_config or GateConfig.default()
        self.pipeline = MoraWeavePipeline(self.gate_config)
        self.teacher_policy = teacher_policy or DelayedTeacherPolicy()
        self.window_ms = window_ms
        self.overlap_ms = overlap_ms
        self.relisten_budget_ms = relisten_budget_ms

    def _cache_key(
        self,
        *,
        namespace: str,
        adapter: ASRAdapter,
        request: DecodeRequest,
        audio_sha256: str,
        context: str = "",
        calibration_digest: str | None = None,
    ) -> RuntimeCacheKey:
        return RuntimeCacheKey.create(
            namespace=namespace,
            audio_sha256=audio_sha256,
            start_ms=request.start_ms or 0,
            end_ms=request.end_ms or 1,
            adapter=adapter.name,
            model=_adapter_model(adapter),
            language=request.language,
            beam_size=request.beam_size,
            hypotheses=request.hypotheses,
            initial_prompt=request.initial_prompt,
            hotwords=request.hotwords,
            context=context,
            calibration_digest=calibration_digest,
        )

    def _decode(
        self,
        adapter: ASRAdapter,
        request: DecodeRequest,
        *,
        namespace: str,
        audio_sha256: str,
        context: str = "",
        calibration_digest: str | None = None,
    ) -> tuple[list[CandidateEvidence], bool]:
        key = self._cache_key(
            namespace=namespace,
            adapter=adapter,
            request=request,
            audio_sha256=audio_sha256,
            context=context,
            calibration_digest=calibration_digest,
        )
        if self.cache is not None:
            cached = self.cache.get_candidates(key)
            if cached is not None:
                return cached, True
        decoded = _scoped_candidates(
            adapter.decode(request),
            namespace=namespace,
            start_ms=request.start_ms or 0,
            end_ms=request.end_ms or 1,
        )
        if not decoded:
            raise RuntimeError(f"{adapter.name} returned no candidates")
        if self.cache is not None:
            self.cache.put_candidates(key, decoded)
        return decoded, False

    def _teacher_result(
        self,
        candidates: list[CandidateEvidence],
        *,
        request: DecodeRequest,
        audio_sha256: str,
        context: str,
        calibration_digest: str | None,
    ) -> tuple[TeacherResult | None, bool]:
        if self.teacher is None:
            return None, False
        lattice_context = json.dumps(
            [{"id": row.candidate_id, "text": row.text} for row in candidates],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        key = RuntimeCacheKey.create(
            namespace="teacher-rank",
            audio_sha256=audio_sha256,
            start_ms=request.start_ms or 0,
            end_ms=request.end_ms or 1,
            adapter="local-teacher",
            model=self.teacher.model,
            language=request.language,
            beam_size=request.beam_size,
            hypotheses=request.hypotheses,
            initial_prompt=request.initial_prompt,
            hotwords=request.hotwords,
            context=context + "\n" + lattice_context,
            calibration_digest=calibration_digest,
        )
        if self.cache is not None:
            cached = self.cache.get_teacher(key)
            if cached is not None:
                return (
                    TeacherResult(
                        probabilities=cached.probabilities,
                        model=cached.model,
                        endpoint_origin="cache",
                        protocol=cached.protocol,
                        entropy=cached.entropy,
                        abstained=cached.abstained,
                    ),
                    True,
                )
        result = self.teacher.probabilities(candidates, context=context)
        if self.cache is not None:
            self.cache.put_teacher(
                key,
                TeacherCacheEntry(
                    probabilities=result.probabilities,
                    abstained=result.abstained,
                    entropy=result.entropy,
                    model=result.model,
                    protocol=result.protocol,
                ),
            )
        return result, False

    @staticmethod
    def _attach_teacher(
        candidates: list[CandidateEvidence], result: TeacherResult
    ) -> list[CandidateEvidence]:
        return [
            replace(candidate, teacher=result.probabilities.get(candidate.candidate_id))
            for candidate in candidates
        ]

    def _transcribe_window(
        self,
        audio_path: str,
        window: Window,
        *,
        audio_sha256: str,
        language: str | None,
        initial_prompt: str | None,
        hotwords: tuple[str, ...],
        context: str,
    ) -> LongformSegment:
        cache_hits: list[str] = []
        base_request = DecodeRequest(
            audio_path=audio_path,
            language=language,
            beam_size=5,
            hypotheses=5,
            start_ms=window.start_ms,
            end_ms=window.end_ms,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
        )
        candidates, hit = self._decode(
            self.base_adapter,
            base_request,
            namespace="base-window",
            audio_sha256=audio_sha256,
            context=context,
        )
        if hit:
            cache_hits.append("base-window")
        ranked = gate_candidates(candidates, self.gate_config)

        relisten_requests = plan_relisten(
            ranked,
            segment_start_ms=window.start_ms,
            segment_end_ms=window.end_ms,
            max_total_ms=self.relisten_budget_ms,
        )
        additional: list[CandidateEvidence] = []
        for relisten_index, relisten in enumerate(relisten_requests):
            request = DecodeRequest(
                audio_path=audio_path,
                language=language,
                beam_size=relisten.beam_size,
                hypotheses=relisten.hypotheses,
                start_ms=relisten.span.start_ms,
                end_ms=relisten.span.end_ms,
                initial_prompt=initial_prompt,
                hotwords=hotwords,
            )
            rows, hit = self._decode(
                self.base_adapter,
                request,
                namespace=f"whisper-relisten-{relisten_index:02d}",
                audio_sha256=audio_sha256,
                context=context,
                calibration_digest=ranked[0].gate.calibration_digest,
            )
            additional.extend(rows)
            if hit:
                cache_hits.append("whisper-relisten")

        if self.second_ear is not None and ranked[0].gate.needs_relisten:
            requests = relisten_requests or [
                type("Fallback", (), {"span": type("Span", (), {"start_ms": window.start_ms, "end_ms": window.end_ms})()})()
            ]
            for second_index, relisten in enumerate(requests):
                request = DecodeRequest(
                    audio_path=audio_path,
                    language=language,
                    beam_size=1,
                    hypotheses=1,
                    start_ms=relisten.span.start_ms,
                    end_ms=relisten.span.end_ms,
                    initial_prompt=initial_prompt,
                    hotwords=hotwords,
                    return_timestamps=True,
                )
                rows, hit = self._decode(
                    self.second_ear,
                    request,
                    namespace=f"qwen-second-ear-{second_index:02d}",
                    audio_sha256=audio_sha256,
                    context=context,
                    calibration_digest=ranked[0].gate.calibration_digest,
                )
                additional.extend(rows)
                if hit:
                    cache_hits.append("qwen-second-ear")

        if additional:
            candidates = merge_relisten_candidates(candidates, additional)
            ranked = gate_candidates(candidates, self.gate_config)

        teacher_result: TeacherResult | None = None
        teacher_cache_hit = False
        if self.teacher is not None and self.teacher_policy.should_query(ranked):
            teacher_result, teacher_cache_hit = self._teacher_result(
                candidates,
                request=base_request,
                audio_sha256=audio_sha256,
                context=context,
                calibration_digest=ranked[0].gate.calibration_digest,
            )
            if teacher_cache_hit:
                cache_hits.append("teacher-rank")
            if teacher_result is not None and not teacher_result.abstained:
                candidates = self._attach_teacher(candidates, teacher_result)
                ranked = gate_candidates(candidates, self.gate_config)

        observed = ObservedTranscript.create(
            selected=ranked[0],
            ranked=ranked,
            uncertainty_spans=[
                {
                    "startMs": request.span.start_ms,
                    "endMs": request.span.end_ms,
                    "reasons": list(request.span.reasons),
                    "priority": request.span.priority,
                }
                for request in relisten_requests
            ],
            source_audio_sha256=audio_sha256,
        )
        observed.verify()

        if teacher_result is not None and not teacher_result.abstained:
            order = sorted(
                teacher_result.probabilities,
                key=lambda candidate_id: (
                    -teacher_result.probabilities[candidate_id],
                    candidate_id,
                ),
            )
            normalized = self.pipeline.normalize_rank_only(observed, order)
        else:
            normalized = NormalizedTranscript.attach(
                observed,
                text=observed.text,
                mode="deterministic",
            )

        diagnostics = {
            **evidence_summary(ranked),
            "candidateCount": len(candidates),
            "relistenCount": len(relisten_requests),
            "secondEarUsed": self.second_ear is not None and ranked[0].gate.needs_relisten,
            "teacherUsed": teacher_result is not None,
            "teacherAbstained": teacher_result.abstained if teacher_result is not None else None,
            "teacherCacheHit": teacher_cache_hit,
        }
        return LongformSegment(
            window=window,
            observed=observed,
            normalized=normalized,
            diagnostics=diagnostics,
            cache_hits=tuple(sorted(set(cache_hits))),
        )

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        duration_ms: int | None = None,
        language: str | None = "ja",
        initial_prompt: str | None = None,
        hotwords: Iterable[str] = (),
        context: str = "",
    ) -> LongformResult:
        source = Path(audio_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        audio_sha256 = sha256_file(source)
        duration = duration_ms or probe_duration_ms(source)
        windows = plan_windows(
            duration,
            window_ms=self.window_ms,
            overlap_ms=self.overlap_ms,
        )
        segments = tuple(
            self._transcribe_window(
                str(source),
                window,
                audio_sha256=audio_sha256,
                language=None if language in {None, "", "auto"} else language,
                initial_prompt=initial_prompt,
                hotwords=tuple(hotwords),
                context=context,
            )
            for window in windows
        )
        observed_text = ""
        normalized_text = ""
        for segment in segments:
            observed_text = stitch_text(observed_text, segment.observed.text)
            normalized_text = stitch_text(normalized_text, segment.normalized.text)
        payload = {
            "audioSha256": audio_sha256,
            "durationMs": duration,
            "observedText": observed_text,
            "normalizedText": normalized_text,
            "segments": segments,
        }
        return LongformResult(
            source_path=str(source),
            source_audio_sha256=audio_sha256,
            duration_ms=duration,
            observed_text=observed_text,
            normalized_text=normalized_text,
            segments=segments,
            evidence_sha256=sha256_json(payload),
            diagnostics={
                "windowCount": len(windows),
                "cacheHitCount": sum(len(segment.cache_hits) for segment in segments),
                "provisionalWindowCount": sum(segment.observed.decision == "provisional" for segment in segments),
                "secondEarWindowCount": sum(bool(segment.diagnostics["secondEarUsed"]) for segment in segments),
                "teacherAbstentionCount": sum(segment.diagnostics["teacherAbstained"] is True for segment in segments),
            },
        )
