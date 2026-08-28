from __future__ import annotations

import argparse
import json
from pathlib import Path

from .engine import EngineConfig, FasterWhisperEngine
from .local_llm import OllamaNormalizer
from .pipeline import PipelineConfig, transcribe_file
from .quality import QualityThresholds

SUPPORTED_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus",
    ".mp4", ".mkv", ".webm", ".mov", ".avi", ".wma",
}
ALL_FORMATS = {"json", "txt", "observed-txt", "md", "srt", "vtt", "tsv", "words-jsonl"}


def _discover_inputs(values: list[str], recursive: bool) -> list[Path]:
    discovered: list[Path] = []
    for value in values:
        path = Path(value).expanduser()
        if path.is_file():
            discovered.append(path.resolve())
            continue
        if path.is_dir():
            iterator = path.rglob("*") if recursive else path.glob("*")
            discovered.extend(item.resolve() for item in iterator if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS)
            continue
        raise FileNotFoundError(path)
    unique = sorted(dict.fromkeys(discovered))
    if not unique:
        raise FileNotFoundError("no supported audio or video files were found")
    return unique


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jtranscribe",
        description="Local-first complete Japanese transcription with timestamps, subtitles and audit evidence.",
    )
    parser.add_argument("inputs", nargs="+", help="audio/video files or directories")
    parser.add_argument("-o", "--output-dir", default="transcripts")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-source-path", action="store_true", help="include absolute input path in JSON/Markdown metadata")
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-type", default="default")
    parser.add_argument("--language", default="ja", help="ja or auto")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--best-of", type=int, default=5)
    parser.add_argument("--vad", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--word-timestamps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--condition-on-previous-text", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--initial-prompt")
    parser.add_argument("--hotwords")
    parser.add_argument("--hotwords-file")
    parser.add_argument("--rttm", help="optional RTTM speaker-diarization file")
    parser.add_argument("--mora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pyopenjtalk", action="store_true", help="add readings/mora for kanji words")
    parser.add_argument("--ollama-model", help="optional local model for guarded readability normalization")
    parser.add_argument("--ollama-endpoint", default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--context", default="")
    parser.add_argument("--formats", default="all", help="comma-separated formats or all")
    parser.add_argument("--min-avg-logprob", type=float, default=-1.0)
    parser.add_argument("--max-no-speech-prob", type=float, default=0.6)
    parser.add_argument("--max-compression-ratio", type=float, default=2.4)
    parser.add_argument("--min-word-probability", type=float, default=0.45)
    parser.add_argument("--log-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    formats = ALL_FORMATS if args.formats == "all" else {item.strip() for item in args.formats.split(",") if item.strip()}
    unknown = formats - ALL_FORMATS
    if unknown:
        parser.error(f"unknown formats: {', '.join(sorted(unknown))}")

    hotwords = args.hotwords
    if args.hotwords_file:
        file_text = Path(args.hotwords_file).read_text(encoding="utf-8").strip()
        hotwords = "、".join(filter(None, [hotwords, file_text]))

    engine = FasterWhisperEngine(
        EngineConfig(
            model=args.model,
            device=args.device,
            compute_type=args.compute_type,
            language=None if args.language == "auto" else args.language,
            beam_size=args.beam_size,
            best_of=args.best_of,
            vad_filter=args.vad,
            word_timestamps=args.word_timestamps,
            condition_on_previous_text=args.condition_on_previous_text,
            initial_prompt=args.initial_prompt,
            hotwords=hotwords,
            log_progress=args.log_progress,
        )
    )

    normalizer = None
    if args.ollama_model:
        normalizer = OllamaNormalizer(model=args.ollama_model, endpoint=args.ollama_endpoint)

    inputs = _discover_inputs(args.inputs, args.recursive)
    output_dir = Path(args.output_dir).expanduser().resolve()
    results: list[dict[str, object]] = []

    for source in inputs:
        source_output_dir = output_dir / source.stem if len(inputs) > 1 else output_dir
        try:
            _, outputs = transcribe_file(
                source,
                engine=engine,
                config=PipelineConfig(
                    output_dir=source_output_dir,
                    formats=set(formats),
                    overwrite=args.overwrite,
                    rttm_path=Path(args.rttm).resolve() if args.rttm else None,
                    annotate_mora=args.mora,
                    use_pyopenjtalk=args.pyopenjtalk,
                    quality=QualityThresholds(
                        min_avg_logprob=args.min_avg_logprob,
                        max_no_speech_prob=args.max_no_speech_prob,
                        max_compression_ratio=args.max_compression_ratio,
                        min_word_probability=args.min_word_probability,
                    ),
                    local_normalizer=normalizer,
                    normalization_context=args.context,
                    include_source_path=args.include_source_path,
                ),
            )
            results.append({"input": str(source), "outputs": outputs, "status": "ok"})
        except Exception as exc:
            results.append({"input": str(source), "status": "error", "error": str(exc)})

    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if all(result["status"] == "ok" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
