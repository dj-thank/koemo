from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .adapters import FasterWhisperAdapter, Qwen3ASRAdapter
from .local_teacher import LocalTeacherClient, OpenAICompatibleTeacherClient
from .longform_v2 import FrontierLongformTranscriber, LongformResult
from .runtime_cache import RuntimeEvidenceCache


def _timecode(milliseconds: int, *, vtt: bool = False) -> str:
    milliseconds = max(0, int(milliseconds))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    separator = "." if vtt else ","
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{millis:03d}"


def _render_subtitles(result: LongformResult, *, normalized: bool, vtt: bool) -> str:
    lines: list[str] = ["WEBVTT", ""] if vtt else []
    for index, segment in enumerate(result.segments, 1):
        text = segment.normalized.text if normalized else segment.observed.text
        if not vtt:
            lines.append(str(index))
        lines.append(
            f"{_timecode(segment.window.start_ms, vtt=vtt)} --> "
            f"{_timecode(segment.window.end_ms, vtt=vtt)}"
        )
        lines.extend([text.strip(), ""])
    return "\n".join(lines).rstrip() + "\n"


def _atomic_write(path: Path, text: str, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def write_outputs(
    result: LongformResult,
    output_dir: str | Path,
    *,
    source_name: str,
    overwrite: bool = False,
    include_source_path: bool = False,
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = Path(source_name).stem
    payload: dict[str, Any] = result.as_dict()
    if not include_source_path:
        payload["source_path"] = Path(source_name).name
    payload["contract"] = {
        "observedImmutable": True,
        "observedEvidenceSha256": result.evidence_sha256,
        "normalizationSeparate": True,
    }
    paths = {
        "json": output / f"{stem}.moraweave.json",
        "observed": output / f"{stem}.observed.txt",
        "normalized": output / f"{stem}.txt",
        "srt": output / f"{stem}.srt",
        "vtt": output / f"{stem}.vtt",
    }
    _atomic_write(
        paths["json"],
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        overwrite=overwrite,
    )
    _atomic_write(paths["observed"], result.observed_text.rstrip() + "\n", overwrite=overwrite)
    _atomic_write(paths["normalized"], result.normalized_text.rstrip() + "\n", overwrite=overwrite)
    _atomic_write(paths["srt"], _render_subtitles(result, normalized=True, vtt=False), overwrite=overwrite)
    _atomic_write(paths["vtt"], _render_subtitles(result, normalized=True, vtt=True), overwrite=overwrite)
    return {name: str(path) for name, path in paths.items()}


def _hotwords(args: argparse.Namespace) -> tuple[str, ...]:
    values: list[str] = []
    if args.hotwords:
        values.extend(item.strip() for item in args.hotwords.split(",") if item.strip())
    if args.hotwords_file:
        text = Path(args.hotwords_file).read_text(encoding="utf-8")
        values.extend(
            item.strip()
            for line in text.splitlines()
            for item in line.replace("、", ",").split(",")
            if item.strip()
        )
    return tuple(dict.fromkeys(values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moraweave-transcribe",
        description="Calibrated, selective and cache-aware complete Japanese transcription.",
    )
    parser.add_argument("audio")
    parser.add_argument("-o", "--output-dir", default="transcripts")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-source-path", action="store_true")
    parser.add_argument("--duration-ms", type=int, help="explicit duration for controlled/offline runs")
    parser.add_argument("--language", default="ja", help="ja or auto")
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-type", default="default")
    parser.add_argument("--window-ms", type=int, default=28_000)
    parser.add_argument("--overlap-ms", type=int, default=1_200)
    parser.add_argument("--initial-prompt")
    parser.add_argument("--hotwords")
    parser.add_argument("--hotwords-file")
    parser.add_argument("--context", default="")
    parser.add_argument("--cache", default=".moraweave/runtime-evidence-v2.sqlite3")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--qwen-second-ear", action="store_true")
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--qwen-device-map", default="cuda:0")
    parser.add_argument("--qwen-dtype", default="float16")
    parser.add_argument("--teacher-model")
    parser.add_argument("--teacher-protocol", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--teacher-endpoint")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = Path(args.audio).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    base = FasterWhisperAdapter(
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
    )
    second_ear = None
    if args.qwen_second_ear:
        second_ear = Qwen3ASRAdapter(
            model=args.qwen_model,
            dtype=args.qwen_dtype,
            device_map=args.qwen_device_map,
            max_inference_batch_size=1,
            return_timestamps=True,
        )

    teacher = None
    if args.teacher_model:
        if args.teacher_protocol == "ollama":
            teacher = LocalTeacherClient(
                model=args.teacher_model,
                endpoint=args.teacher_endpoint or "http://127.0.0.1:11434/api/chat",
            )
        else:
            teacher = OpenAICompatibleTeacherClient(
                model=args.teacher_model,
                endpoint=args.teacher_endpoint or "http://127.0.0.1:8000/v1/chat/completions",
                preserve_thinking=False,
            )

    cache = None if args.no_cache else RuntimeEvidenceCache(args.cache)
    try:
        transcriber = FrontierLongformTranscriber(
            base,
            second_ear=second_ear,
            teacher=teacher,
            cache=cache,
            window_ms=args.window_ms,
            overlap_ms=args.overlap_ms,
        )
        result = transcriber.transcribe(
            source,
            duration_ms=args.duration_ms,
            language=None if args.language == "auto" else args.language,
            initial_prompt=args.initial_prompt,
            hotwords=_hotwords(args),
            context=args.context,
        )
        outputs = write_outputs(
            result,
            args.output_dir,
            source_name=source.name,
            overwrite=args.overwrite,
            include_source_path=args.include_source_path,
        )
    finally:
        if cache is not None:
            cache.close()

    print(json.dumps({"status": "ok", "outputs": outputs, "diagnostics": result.diagnostics}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
