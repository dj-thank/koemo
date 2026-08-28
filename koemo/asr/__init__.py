"""Mora-aware ASR contracts and incremental pipeline utilities."""

from .mora import (
    collapse_ctc_path,
    merge_char_ctc_to_mora,
    mora_count,
    mora_units_from_reading,
    normalize_kana,
    split_mora,
)
from .schema import (
    CTCCharUnit,
    HypothesisFeatures,
    MoraUnit,
    RankedHypothesis,
    TextSpan,
    TimeSpan,
    TranscriptHypothesis,
    TranscriptState,
    UnitKind,
    UnitSource,
)
from .scoring import (
    LLMRankVote,
    ScoreWeights,
    attach_llm_rank_only,
    attach_normalized_transcript,
    rank_acoustic_hypotheses,
    select_observed_transcript,
)
from .whisper_nbest import (
    WhisperNBestItem,
    as_transcript_hypotheses,
    decode_nbest_window,
    resolve_faster_whisper_suppress_tokens,
)

__all__ = [
    "CTCCharUnit",
    "HypothesisFeatures",
    "LLMRankVote",
    "MoraUnit",
    "RankedHypothesis",
    "ScoreWeights",
    "TextSpan",
    "TimeSpan",
    "TranscriptHypothesis",
    "TranscriptState",
    "UnitKind",
    "UnitSource",
    "WhisperNBestItem",
    "as_transcript_hypotheses",
    "attach_llm_rank_only",
    "attach_normalized_transcript",
    "collapse_ctc_path",
    "decode_nbest_window",
    "merge_char_ctc_to_mora",
    "mora_count",
    "mora_units_from_reading",
    "normalize_kana",
    "rank_acoustic_hypotheses",
    "resolve_faster_whisper_suppress_tokens",
    "select_observed_transcript",
    "split_mora",
]
