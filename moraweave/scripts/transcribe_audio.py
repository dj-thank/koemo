from __future__ import annotations

import argparse
import json
from pathlib import Path

from moraweave.longform import LongFormConfig, transcribe_longform


def main() -> int:
    parser = argparse.ArgumentParser(description="MoraWeave complete Japanese transcription")
    parser.add_argument("audio")
    parser.add_argument("--output-dir", default="transcripts")
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-type", default="default")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--nbest-beam-size", type=int, default=12)
    parser.add_argument("--nbest-hypotheses", type=int, default=8)
    parser.add_argument("--initial-prompt")
    parser.add_argument("--hotwords", default="")
    parser.add_argument("--memory-database")
    parser.add_argument("--memory-namespace", default="public-ja")
    parser.add_argument("--teacher-model")
    parser.add_argument("--teacher-endpoint", default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--teacher-cache")
    parser.add_argument("--qwen-second-ear", action="store_true")
    parser.add_argument("--no-vad", action="store_true")
    args = parser.parse_args()

    config = LongFormConfig(
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        nbest_beam_size=args.nbest_beam_size,
        nbest_hypotheses=args.nbest_hypotheses,
        initial_prompt=args.initial_prompt,
        hotwords=tuple(item.strip() for item in args.hotwords.split(",") if item.strip()),
        vad_filter=not args.no_vad,
        memory_database=args.memory_database,
        memory_namespace=args.memory_namespace,
        teacher_model=args.teacher_model,
        teacher_endpoint=args.teacher_endpoint,
        teacher_cache=args.teacher_cache,
        qwen_second_ear=args.qwen_second_ear,
    )
    document = transcribe_longform(
        args.audio,
        output_dir=Path(args.output_dir),
        config=config,
    )
    print(json.dumps(document["outputs"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
