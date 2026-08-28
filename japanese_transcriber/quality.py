from __future__ import annotations

from dataclasses import dataclass

from .types import Segment


@dataclass(slots=True)
class QualityThresholds:
    min_avg_logprob: float = -1.0
    max_no_speech_prob: float = 0.6
    max_compression_ratio: float = 2.4
    min_word_probability: float = 0.45


def annotate_segment(segment: Segment, thresholds: QualityThresholds) -> Segment:
    reasons: list[str] = []
    if segment.avg_logprob is not None and segment.avg_logprob < thresholds.min_avg_logprob:
        reasons.append("low-average-logprob")
    if segment.no_speech_prob is not None and segment.no_speech_prob > thresholds.max_no_speech_prob:
        reasons.append("high-no-speech-probability")
    if segment.compression_ratio is not None and segment.compression_ratio > thresholds.max_compression_ratio:
        reasons.append("high-compression-ratio")

    low_words = [
        word
        for word in segment.words
        if word.probability is not None and word.probability < thresholds.min_word_probability
    ]
    if low_words:
        reasons.append("low-word-probability")

    segment.uncertainty_reasons = reasons
    return segment


def summarize_quality(segments: list[Segment]) -> dict[str, object]:
    uncertain = [segment for segment in segments if segment.uncertainty_reasons]
    words = [word for segment in segments for word in segment.words]
    probabilities = [word.probability for word in words if word.probability is not None]
    return {
        "segmentCount": len(segments),
        "uncertainSegmentCount": len(uncertain),
        "uncertainSegmentIds": [segment.id for segment in uncertain],
        "wordCount": len(words),
        "meanWordProbability": sum(probabilities) / len(probabilities) if probabilities else None,
    }
