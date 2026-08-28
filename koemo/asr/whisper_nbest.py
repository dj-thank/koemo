"""Thin N-best adapter over faster-whisper's CTranslate2 model.

``faster-whisper`` exposes one best sequence in its public ``Segment`` objects,
but the underlying ``ctranslate2.models.Whisper.generate`` API can return
multiple hypotheses and their scores.  This module deliberately wraps only that
low-level window decode.  VAD/chunking/timestamp splitting remain owned by the
existing transcription pipeline so we do not silently fork upstream behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Callable, Iterable, Sequence

from .schema import HypothesisFeatures, TranscriptHypothesis, UnitSource


@dataclass(frozen=True, slots=True)
class WhisperNBestItem:
    candidate_id: str
    token_ids: tuple[int, ...]
    sequence_score: float
    average_logprob: float
    no_speech_probability: float | None
    text: str | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id is required")
        if not isfinite(self.sequence_score) or not isfinite(self.average_logprob):
            raise ValueError("Whisper scores must be finite")
        if self.no_speech_probability is not None and not 0.0 <= self.no_speech_probability <= 1.0:
            raise ValueError("no_speech_probability must be in [0, 1]")

    def to_transcript_hypothesis(self) -> TranscriptHypothesis:
        if self.text is None:
            raise ValueError("text decoder was not supplied for this N-best item")
        return TranscriptHypothesis(
            candidate_id=self.candidate_id,
            text=self.text,
            token_ids=self.token_ids,
            features=HypothesisFeatures(
                whisper_logprob=self.average_logprob,
                no_speech_probability=self.no_speech_probability,
            ),
            source=UnitSource.WHISPER,
        )


def resolve_faster_whisper_suppress_tokens(
    tokenizer: Any,
    suppress_tokens: Sequence[int] = (-1,),
) -> tuple[int, ...]:
    """Resolve faster-whisper's ``-1`` sentinel before calling CTranslate2.

    The public faster-whisper option ``-1`` means the tokenizer's default
    non-speech set.  CTranslate2 itself does not define that sentinel, so a
    low-level adapter must expand it explicitly.  Control tokens are appended
    exactly as faster-whisper does.
    """

    values = [int(token) for token in suppress_tokens]
    if -1 in values:
        values = [token for token in values if token >= 0]
        values.extend(int(token) for token in tokenizer.non_speech_tokens)
    elif any(token < 0 for token in values):
        raise ValueError("negative suppression tokens other than -1 are invalid")

    required_names = (
        "transcribe",
        "translate",
        "sot",
        "sot_prev",
        "sot_lm",
        "no_speech",
    )
    missing = [name for name in required_names if not hasattr(tokenizer, name)]
    if missing:
        raise TypeError(f"tokenizer is missing suppression attributes: {missing}")
    values.extend(int(getattr(tokenizer, name)) for name in required_names)
    return tuple(sorted(set(values)))


def _unwrap_ctranslate2_whisper(model: Any) -> Any:
    """Accept either ``faster_whisper.WhisperModel`` or raw CT2 Whisper."""

    raw = getattr(model, "model", model)
    if not callable(getattr(raw, "generate", None)):
        raise TypeError("model must expose CTranslate2 Whisper.generate")
    return raw


def decode_nbest_window(
    model: Any,
    encoded_features: Any,
    prompt_tokens: Sequence[int],
    *,
    num_hypotheses: int = 5,
    beam_size: int = 5,
    patience: float = 1.0,
    length_penalty: float = 1.0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    max_length: int = 448,
    suppress_blank: bool = True,
    suppress_tokens: Sequence[int] | None = None,
    max_initial_timestamp_index: int = 50,
    window_id: str = "0000",
    decode_tokens: Callable[[Sequence[int]], str] | None = None,
    tokenizer: Any | None = None,
) -> tuple[WhisperNBestItem, ...]:
    """Decode one encoded Whisper window into deterministic beam N-best.

    CTranslate2 returns a length-penalized sequence score.  Following
    faster-whisper's score handling, this function reconstructs cumulative log
    probability and then divides by ``sequence_length + 1`` to produce an
    average log probability suitable for later cross-candidate calibration.
    """

    if num_hypotheses < 1:
        raise ValueError("num_hypotheses must be >= 1")
    if beam_size < 1:
        raise ValueError("beam_size must be >= 1")
    effective_beam_size = max(beam_size, num_hypotheses)

    if tokenizer is not None:
        requested_suppress_tokens = (-1,) if suppress_tokens is None else suppress_tokens
        resolved_suppress_tokens = resolve_faster_whisper_suppress_tokens(
            tokenizer, requested_suppress_tokens
        )
        if decode_tokens is None:
            decode_tokens = tokenizer.decode
    else:
        requested_suppress_tokens = () if suppress_tokens is None else suppress_tokens
        if any(int(token) < 0 for token in requested_suppress_tokens):
            raise ValueError(
                "negative suppression sentinels require a faster-whisper tokenizer"
            )
        resolved_suppress_tokens = tuple(int(token) for token in requested_suppress_tokens)

    raw_model = _unwrap_ctranslate2_whisper(model)
    results = raw_model.generate(
        encoded_features,
        [list(prompt_tokens)],
        beam_size=effective_beam_size,
        patience=patience,
        num_hypotheses=num_hypotheses,
        length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        max_length=max_length,
        return_scores=True,
        return_no_speech_prob=True,
        max_initial_timestamp_index=max_initial_timestamp_index,
        suppress_blank=suppress_blank,
        suppress_tokens=list(resolved_suppress_tokens),
        sampling_topk=1,
        sampling_temperature=0.0,
    )
    if len(results) != 1:
        raise RuntimeError(f"expected one window result, received {len(results)}")

    result = results[0]
    sequences = tuple(tuple(int(token) for token in seq) for seq in result.sequences_ids)
    scores = tuple(float(score) for score in result.scores)
    if len(sequences) != len(scores):
        raise RuntimeError("CTranslate2 returned mismatched sequences and scores")
    if not sequences:
        raise RuntimeError("CTranslate2 returned no hypotheses")
    if len(sequences) < num_hypotheses:
        # Beam search can legitimately finish fewer unique hypotheses.  Preserve
        # what exists instead of fabricating candidates.
        num_hypotheses = len(sequences)

    raw_no_speech = getattr(result, "no_speech_prob", None)
    no_speech = None if raw_no_speech is None else float(raw_no_speech)

    items: list[WhisperNBestItem] = []
    for index, (token_ids, score) in enumerate(zip(sequences, scores)):
        if index >= num_hypotheses:
            break
        sequence_length = max(1, len(token_ids))
        cumulative_logprob = score * (sequence_length**length_penalty)
        average_logprob = cumulative_logprob / (sequence_length + 1)
        text = None if decode_tokens is None else decode_tokens(token_ids)
        items.append(
            WhisperNBestItem(
                candidate_id=f"w{window_id}-h{index:02d}",
                token_ids=token_ids,
                sequence_score=score,
                average_logprob=average_logprob,
                no_speech_probability=no_speech,
                text=text,
            )
        )
    return tuple(items)


def as_transcript_hypotheses(
    items: Iterable[WhisperNBestItem],
) -> tuple[TranscriptHypothesis, ...]:
    return tuple(item.to_transcript_hypothesis() for item in items)
