#!/usr/bin/env python3
"""Expose CTranslate2 Whisper N-best hypotheses from faster-whisper.

The adapter intentionally handles one utterance (at most one Whisper window). Long
recordings must be segmented before calling it so hypotheses from unrelated windows
are never combined into a false global N-best list.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class AsrCandidate:
    id: int
    text: str
    token_ids: list[int]
    whisper_score: float


def _score_list(result: Any, count: int) -> list[float]:
    scores = list(getattr(result, "scores", ()) or ())
    if not scores:
        return [0.0] * count
    if len(scores) != count:
        raise RuntimeError("CTranslate2 returned a different number of scores and hypotheses")
    return [float(score) for score in scores]


def decode_generation_result(result: Any, tokenizer: Any) -> list[AsrCandidate]:
    sequences: Sequence[Sequence[int]] = getattr(result, "sequences_ids", ()) or ()
    scores = _score_list(result, len(sequences))
    timestamp_begin = getattr(tokenizer, "timestamp_begin", None)
    candidates: list[AsrCandidate] = []

    for index, (sequence, score) in enumerate(zip(sequences, scores, strict=True)):
        token_ids = [int(token) for token in sequence]
        if timestamp_begin is not None:
            text_tokens = [token for token in token_ids if token < int(timestamp_begin)]
        else:
            text_tokens = token_ids
        text = tokenizer.decode(text_tokens).strip()
        candidates.append(
            AsrCandidate(
                id=index,
                text=text,
                token_ids=token_ids,
                whisper_score=score,
            )
        )

    if not candidates:
        raise RuntimeError("CTranslate2 returned no hypotheses")
    return candidates


def transcribe_nbest(
    audio_path: str | Path,
    *,
    model_name: str = "small",
    device: str = "auto",
    compute_type: str = "default",
    language: str = "ja",
    beam_size: int = 8,
    nbest: int = 5,
) -> dict[str, Any]:
    try:
        import numpy as np
        from faster_whisper import WhisperModel
        from faster_whisper.audio import decode_audio
        from faster_whisper.tokenizer import Tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Install faster-whisper and numpy before running N-best inference"
        ) from exc

    audio_path = Path(audio_path)
    audio_bytes = audio_path.read_bytes()
    audio = decode_audio(str(audio_path), sampling_rate=16_000)
    duration_seconds = len(audio) / 16_000
    if duration_seconds > 30.0:
        raise ValueError("audio exceeds one Whisper window; segment it before N-best inference")

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    tokenizer = Tokenizer(
        model.hf_tokenizer,
        model.model.is_multilingual,
        task="transcribe",
        language=language,
    )

    features = model.feature_extractor(audio)
    if isinstance(features, tuple):
        features = features[0]
    features = np.asarray(features)
    if features.ndim == 2:
        features = np.expand_dims(features, axis=0)

    encoder_output = model.encode(features)
    prompt = list(tokenizer.sot_sequence)
    generated = model.model.generate(
        encoder_output,
        [prompt],
        beam_size=beam_size,
        num_hypotheses=nbest,
        return_scores=True,
        return_no_speech_prob=True,
        sampling_temperature=0.0,
    )
    if len(generated) != 1:
        raise RuntimeError("expected exactly one generated utterance")

    candidates = decode_generation_result(generated[0], tokenizer)
    return {
        "schemaVersion": "0.3.0",
        "engine": "faster-whisper-ctranslate2-nbest",
        "model": model_name,
        "language": language,
        "durationSeconds": duration_seconds,
        "audioSha256": hashlib.sha256(audio_bytes).hexdigest(),
        "beamSize": beam_size,
        "requestedHypotheses": nbest,
        "candidates": [asdict(candidate) for candidate in candidates],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--model", default="small")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-type", default="default")
    parser.add_argument("--language", default="ja")
    parser.add_argument("--beam-size", type=int, default=8)
    parser.add_argument("--nbest", type=int, default=5)
    parser.add_argument("--output")
    args = parser.parse_args()

    payload = transcribe_nbest(
        args.audio,
        model_name=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        beam_size=args.beam_size,
        nbest=args.nbest,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
