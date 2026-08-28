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
from .gates import GateConfig, gate_candidates
from .local_teacher import TeacherResult
from .runtime_cache import RuntimeCacheKey, RuntimeEvidenceCache, TeacherCacheEntry


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
class RelistenSpan:
    start_ms: int
    end_ms: int
    priority: float
    reasons: tuple[str, ...]


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
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(source),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        duration = float(completed.stdout.strip())
        if duration > 0:
            return max(1, round(duration * 1000))
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        pass
    if source.suffix.lower() == ".wav":
        with wave.open(str(source), "rb") as stream:
            return max(1, round(stream.getnframes() / stream.getframerate() * 1000))
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
        raise ValueError("invalid window/overlap configuration")
    output: list[Window] = []
    start = 0
    index = 0
    while start < duration_ms:
        end = min(duration_ms, start + window_ms)
        output.append(Window(index, start, end))
        if end == duration_ms:
            break
        start = end - overlap_ms
        index += 1
    return output


def _ascii_boundary(left: str, right: str) -> bool:
    return bool(
        left
        and right
        and left[-1].isascii()
        and right[0].isascii()
        and left[-1].isalnum()
        and right[0].isalnum()
    )


def stitch_text(left: str, right: str, *, maximum_overlap_chars: int = 160) -> str:
    left = str(left or "").strip()
    right = str(right or "").strip()
    if not left:
        return right
    if not right:
        return left
    overlap = 0
    for length in range(min(maximum_overlap_chars, len(left), len(right)), 1, -1):
        if left[-length:] == right[:length]:
            overlap = length
            break
    suffix = right[overlap:]
    if not suffix:
        return left
    if left[-1:] in "、。！？!?" and suffix[:1] == left[-1:]:
        suffix = suffix[1:]
    return left + (" " if _ascii_boundary(left, suffix) else "") + suffix


def _adapter_model(adapter: ASRAdapter) -> str:
    return str(getattr(adapter, "model_name", getattr(adapter, "name", type(adapter).__name__)))


def _scope(
    candidates: Iterable[CandidateEvidence],
    *,
    namespace: str,
    start_ms: int,
    end_ms: int,
) -> list[CandidateEvidence]:
    output: list[CandidateEvidence] = []
    for index, candidate in enumerate(candidates, 1):
        metadata = dict(candidate.metadata)
        support = set(str(value) for value in metadata.get("sourceSupport", []))
        support.add(candidate.evidence_source)
        metadata.update(
            {
                "sourceSupport": sorted(support),
                "decodeNamespace": namespace,
                "decodeStartMs": start_ms,
                "decodeEndMs": end_ms,
            }
        )
        output.append(
            replace(
                candidate,
                candidate_id=f"{namespace}:{start_ms}-{end_ms}:{index:04d}",
                source=candidate.evidence_source,
                metadata=metadata,
            )
        )
    return output


def _strength(candidate: CandidateEvidence) -> tuple[float, float, float]:
    return (
        float(candidate.acoustic) if candidate.acoustic is not None else -1e9,
        float(candidate.mora) if candidate.mora is not None else -1e9,
        float(candidate.avg_logprob) if candidate.avg_logprob is not None else -1e9,
    )


def merge_candidates(
    primary: Iterable[CandidateEvidence], additional: Iterable[CandidateEvidence]
) -> list[CandidateEvidence]:
    by_text: dict[str, CandidateEvidence] = {}
    support: dict[str, set[str]] = {}
    for candidate in [*primary, *additional]:
        support.setdefault(candidate.text, set()).update(
            str(value) for value in candidate.metadata.get("sourceSupport", [])
        )
        support[candidate.text].add(candidate.evidence_source)
        current = by_text.get(candidate.text)
        if current is None or _strength(candidate) > _strength(current):
            by_text[candidate.text] = candidate
    output: list[CandidateEvidence] = []
    for index, text in enumerate(sorted(by_text), 1):
        candidate = by_text[text]
        metadata = dict(candidate.metadata)
        metadata["sourceSupport"] = sorted(support[text])
        output.append(
            replace(
                candidate,
                candidate_id=f"merged:{index:04d}",
                metadata=metadata,
            )
        )
    return output


def _relisten_span(
    window: Window,
    ranked: list,
    *,
    budget_ms: int,
) -> RelistenSpan | None:
    gate = ranked[0].gate
    if not gate.needs_relisten:
        return None
    length = min(budget_ms, window.end_ms - window.start_ms)
    midpoint = (window.start_ms + window.end_ms) // 2
    start = max(window.start_ms, midpoint - length // 2)
    end = min(window.end_ms, start + length)
    start = max(window.start_ms, end - length)
    priority = max(gate.selective_risk, gate.entropy, gate.disagreement)
    return RelistenSpan(start, end, priority, tuple(gate.reasons))


class FrontierLongformTranscriber:
    """Cache-aware long-form decoder with uncertainty-triggered second evidence."""

    def __init__(
        self,
        base_adapter: ASRAdapter,
        *,
        second_ear: ASRAdapter | None = None,
        teacher: TeacherClient | None = None,
        cache: RuntimeEvidenceCache | None = None,
        gate_config: GateConfig | None = None,
        window_ms: int = 28_000,
        overlap_ms: int = 1_200,
        relisten_budget_ms: int = 10_000,
        teacher_posterior_threshold: float = 0.86,
    ) -> None:
        self.base_adapter = base_adapter
        self.second_ear = second_ear
        self.teacher = teacher
        self.cache = cache
        if gate_config is None:
            default = getattr(GateConfig, "default", None)
            gate_config = default() if callable(default) else GateConfig()
        self.gate_config = gate_config
        self.window_ms = window_ms
        self.overlap_ms = overlap_ms
        self.relisten_budget_ms = relisten_budget_ms
        self.teacher_posterior_threshold = teacher_posterior_threshold

    def _key(
        self,
        namespace: str,
        adapter: ASRAdapter,
        request: DecodeRequest,
        audio_sha256: str,
        *,
        context: str,
        calibration_digest: str | None = None,
    ) -> RuntimeCacheKey:
        return RuntimeCacheKey.create(
            namespace=namespace,
            audio_sha256=audio_sha256,
            start_ms=int(request.start_ms or 0),
            end_ms=int(request.end_ms or 1),
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
        context: str,
        calibration_digest: str | None = None,
    ) -> tuple[list[CandidateEvidence], bool]:
        key = self._key(
            namespace,
            adapter,
            request,
            audio_sha256,
            context=context,
            calibration_digest=calibration_digest,
        )
        if self.cache is not None:
            cached = self.cache.get_candidates(key)
            if cached is not None:
                return cached, True
        candidates = _scope(
            adapter.decode(request),
            namespace=namespace,
            start_ms=int(request.start_ms or 0),
            end_ms=int(request.end_ms or 1),
        )
        if not candidates:
            raise RuntimeError(f"{adapter.name} returned no candidates")
        if self.cache is not None:
            self.cache.put_candidates(key, candidates)
        return candidates, False

    def _teacher(
        self,
        candidates: list[CandidateEvidence],
        request: DecodeRequest,
        *,
        audio_sha256: str,
        context: str,
        calibration_digest: str | None,
    ) -> tuple[TeacherResult | None, bool]:
        if self.teacher is None:
            return None, False
        lattice = json.dumps(
            [{"id": row.candidate_id, "text": row.text} for row in candidates],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        key = RuntimeCacheKey.create(
            namespace="teacher-rank",
            audio_sha256=audio_sha256,
            start_ms=int(request.start_ms or 0),
            end_ms=int(request.end_ms or 1),
            adapter="local-teacher",
            model=self.teacher.model,
            language=request.language,
            beam_size=request.beam_size,
            hypotheses=request.hypotheses,
            initial_prompt=request.initial_prompt,
            hotwords=request.hotwords,
            context=context + "\n" + lattice,
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

    def _window(
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
        initial_needs_relisten = ranked[0].gate.needs_relisten
        span = _relisten_span(window, ranked, budget_ms=self.relisten_budget_ms)
        additional: list[CandidateEvidence] = []

        if span is not None:
            request = DecodeRequest(
                audio_path=audio_path,
                language=language,
                beam_size=12,
                hypotheses=8,
                start_ms=span.start_ms,
                end_ms=span.end_ms,
                initial_prompt=initial_prompt,
                hotwords=hotwords,
            )
            rows, hit = self._decode(
                self.base_adapter,
                request,
                namespace="whisper-relisten",
                audio_sha256=audio_sha256,
                context=context,
                calibration_digest=ranked[0].gate.calibration_digest,
            )
            additional.extend(rows)
            if hit:
                cache_hits.append("whisper-relisten")

        second_ear_used = False
        if self.second_ear is not None and span is not None:
            request = DecodeRequest(
                audio_path=audio_path,
                language=language,
                beam_size=1,
                hypotheses=1,
                start_ms=span.start_ms,
                end_ms=span.end_ms,
                initial_prompt=initial_prompt,
                hotwords=hotwords,
                return_timestamps=True,
            )
            rows, hit = self._decode(
                self.second_ear,
                request,
                namespace="qwen-second-ear",
                audio_sha256=audio_sha256,
                context=context,
                calibration_digest=ranked[0].gate.calibration_digest,
            )
            additional.extend(rows)
            second_ear_used = True
            if hit:
                cache_hits.append("qwen-second-ear")

        if additional:
            candidates = merge_candidates(candidates, additional)
            ranked = gate_candidates(candidates, self.gate_config)

        teacher_result: TeacherResult | None = None
        teacher_cache_hit = False
        if self.teacher is not None and (
            ranked[0].gate.needs_relisten
            or ranked[0].posterior < self.teacher_posterior_threshold
        ):
            teacher_result, teacher_cache_hit = self._teacher(
                candidates,
                base_request,
                audio_sha256=audio_sha256,
                context=context,
                calibration_digest=ranked[0].gate.calibration_digest,
            )
            if teacher_cache_hit:
                cache_hits.append("teacher-rank")
            if teacher_result is not None and not teacher_result.abstained:
                candidates = [
                    replace(row, teacher=teacher_result.probabilities.get(row.candidate_id))
                    for row in candidates
                ]
                ranked = gate_candidates(candidates, self.gate_config)

        uncertainty = []
        if span is not None:
            uncertainty.append(
                {
                    "startMs": span.start_ms,
                    "endMs": span.end_ms,
                    "priority": span.priority,
                    "reasons": list(span.reasons),
                }
            )
        observed = ObservedTranscript.create(
            selected=ranked[0],
            ranked=ranked,
            uncertainty_spans=uncertainty,
            source_audio_sha256=audio_sha256,
        )
        observed.verify()

        if teacher_result is not None and not teacher_result.abstained:
            selected_id = max(
                teacher_result.probabilities,
                key=lambda candidate_id: (
                    teacher_result.probabilities[candidate_id],
                    candidate_id,
                ),
            )
            selected = next(row for row in observed.candidates if row.candidate_id == selected_id)
            normalized = NormalizedTranscript.attach(
                observed,
                text=selected.text,
                mode="rank-only",
                selected_candidate_id=selected_id,
            )
        else:
            normalized = NormalizedTranscript.attach(
                observed,
                text=observed.text,
                mode="deterministic",
            )

        gate = ranked[0].gate
        diagnostics = {
            "posterior": ranked[0].posterior,
            "entropy": gate.entropy,
            "disagreement": gate.disagreement,
            "evidenceCoverage": gate.evidence_coverage,
            "selectiveRisk": gate.selective_risk,
            "abstain": gate.abstain,
            "reasons": list(gate.reasons),
            "candidateCount": len(candidates),
            "initialNeedsRelisten": initial_needs_relisten,
            "relistenUsed": span is not None,
            "secondEarUsed": second_ear_used,
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
        duration = int(duration_ms or probe_duration_ms(source))
        windows = plan_windows(duration, window_ms=self.window_ms, overlap_ms=self.overlap_ms)
        normalized_language = None if language in {None, "", "auto"} else language
        hotword_tuple = tuple(str(value) for value in hotwords if str(value))
        segments = tuple(
            self._window(
                str(source),
                window,
                audio_sha256=audio_sha256,
                language=normalized_language,
                initial_prompt=initial_prompt,
                hotwords=hotword_tuple,
                context=context,
            )
            for window in windows
        )
        observed_text = ""
        normalized_text = ""
        for segment in segments:
            observed_text = stitch_text(observed_text, segment.observed.text)
            normalized_text = stitch_text(normalized_text, segment.normalized.text)
        evidence_payload = {
            "audioSha256": audio_sha256,
            "durationMs": duration,
            "observedText": observed_text,
            "normalizedText": normalized_text,
            "segmentEvidence": [segment.observed.evidence_sha256 for segment in segments],
        }
        return LongformResult(
            source_path=str(source),
            source_audio_sha256=audio_sha256,
            duration_ms=duration,
            observed_text=observed_text,
            normalized_text=normalized_text,
            segments=segments,
            evidence_sha256=sha256_json(evidence_payload),
            diagnostics={
                "windowCount": len(windows),
                "cacheHitCount": sum(len(segment.cache_hits) for segment in segments),
                "provisionalWindowCount": sum(segment.observed.decision == "provisional" for segment in segments),
                "relistenWindowCount": sum(bool(segment.diagnostics["relistenUsed"]) for segment in segments),
                "secondEarWindowCount": sum(bool(segment.diagnostics["secondEarUsed"]) for segment in segments),
                "teacherAbstentionCount": sum(segment.diagnostics["teacherAbstained"] is True for segment in segments),
            },
        )
