from __future__ import annotations

import math
import unittest

from koemo.asr import (
    CTCCharUnit,
    HypothesisFeatures,
    LLMRankVote,
    RankedHypothesis,
    TranscriptHypothesis,
    TranscriptState,
    UnitKind,
    attach_llm_rank_only,
    attach_normalized_transcript,
    collapse_ctc_path,
    decode_nbest_window,
    merge_char_ctc_to_mora,
    mora_count,
    normalize_kana,
    resolve_faster_whisper_suppress_tokens,
    select_observed_transcript,
    split_mora,
)


class MoraSegmentationTests(unittest.TestCase):
    def test_normalizes_hiragana_to_katakana(self) -> None:
        self.assertEqual(normalize_kana("きゃっと"), "キャット")

    def test_reading_preserves_surface_script(self) -> None:
        units = merge_char_ctc_to_mora((CTCCharUnit("き"), CTCCharUnit("ゃ")))
        self.assertEqual(units[0].surface, "きゃ")
        self.assertEqual(units[0].reading, "キャ")

    def test_core_japanese_mora_rules(self) -> None:
        cases = {
            "キャット": ("キャ", "ッ", "ト"),
            "スーパー": ("ス", "ー", "パ", "ー"),
            "しんよう": ("シ", "ン", "ヨ", "ウ"),
            "ティッシュ": ("ティ", "ッ", "シュ"),
            "ヴォイス": ("ヴォ", "イ", "ス"),
            "ウィンドウ": ("ウィ", "ン", "ド", "ウ"),
        }
        for reading, expected in cases.items():
            with self.subTest(reading=reading):
                self.assertEqual(split_mora(reading), expected)
                self.assertEqual(mora_count(reading), len(expected))

    def test_char_ctc_merges_small_kana_and_time_spans(self) -> None:
        units = (
            CTCCharUnit("キ", posterior=0.90, start=0.00, end=0.08),
            CTCCharUnit("ャ", posterior=0.81, start=0.08, end=0.14),
            CTCCharUnit("ッ", posterior=0.95, start=0.14, end=0.20),
            CTCCharUnit("ト", posterior=0.99, start=0.20, end=0.30),
        )
        mora_units = merge_char_ctc_to_mora(units)
        self.assertEqual(tuple(item.mora for item in mora_units), ("キャ", "ッ", "ト"))
        self.assertEqual(mora_units[0].kind, UnitKind.MORA)
        self.assertEqual(mora_units[0].time_span.start, 0.0)
        self.assertEqual(mora_units[0].time_span.end, 0.14)
        self.assertAlmostEqual(mora_units[0].posterior, math.sqrt(0.90 * 0.81))
        self.assertEqual(mora_units[0].source_indices, (0, 1))

    def test_boundaries_are_optional_and_not_mora(self) -> None:
        units = (
            CTCCharUnit("コ"),
            CTCCharUnit("エ"),
            CTCCharUnit("モ"),
            CTCCharUnit("。"),
        )
        without = merge_char_ctc_to_mora(units)
        with_boundaries = merge_char_ctc_to_mora(units, keep_boundaries=True)
        self.assertEqual(len(without), 3)
        self.assertEqual(with_boundaries[-1].kind, UnitKind.BOUNDARY)
        self.assertEqual(with_boundaries[-1].mora, "")

    def test_standard_ctc_collapse_respects_blank_separator(self) -> None:
        path = (
            CTCCharUnit("カ", posterior=0.8, start=0.00, end=0.02),
            CTCCharUnit("カ", posterior=0.9, start=0.02, end=0.04),
            CTCCharUnit("", posterior=0.9, start=0.04, end=0.06, is_blank=True),
            CTCCharUnit("カ", posterior=0.7, start=0.06, end=0.08),
        )
        collapsed = collapse_ctc_path(path)
        self.assertEqual(tuple(item.symbol for item in collapsed), ("カ", "カ"))
        self.assertEqual(collapsed[0].start, 0.0)
        self.assertEqual(collapsed[0].end, 0.04)
        self.assertEqual(collapsed[1].start, 0.06)


class ScoringTests(unittest.TestCase):
    def _candidates(self) -> tuple[TranscriptHypothesis, ...]:
        return (
            TranscriptHypothesis(
                candidate_id="h0",
                text="音声認識です",
                features=HypothesisFeatures(
                    whisper_logprob=-0.30,
                    char_ctc_logprob=-0.25,
                    mora_ctc_logprob=-0.20,
                    alignment_quality=0.90,
                    no_speech_probability=0.02,
                    compression_ratio=1.2,
                ),
            ),
            TranscriptHypothesis(
                candidate_id="h1",
                text="音声人識です",
                features=HypothesisFeatures(
                    whisper_logprob=-0.35,
                    char_ctc_logprob=-0.70,
                    mora_ctc_logprob=-0.80,
                    alignment_quality=0.60,
                    no_speech_probability=0.03,
                    compression_ratio=1.2,
                ),
            ),
            TranscriptHypothesis(
                candidate_id="h2",
                text="音声認識でした",
                features=HypothesisFeatures(
                    whisper_logprob=-0.45,
                    char_ctc_logprob=-0.40,
                    mora_ctc_logprob=-0.50,
                    alignment_quality=0.75,
                    no_speech_probability=0.04,
                    compression_ratio=2.8,
                ),
            ),
        )

    def test_stage2_freezes_observed_from_acoustic_evidence(self) -> None:
        state = select_observed_transcript(self._candidates())
        self.assertEqual(state.observed_candidate_id, "h0")
        self.assertEqual(state.observed_transcript, "音声認識です")
        self.assertEqual(state.hypotheses[0].rank, 1)

    def test_llm_rank_only_does_not_replace_observed_text(self) -> None:
        state = select_observed_transcript(self._candidates())
        ranked = attach_llm_rank_only(
            state,
            (
                LLMRankVote("h1", rank=1, confidence=0.95),
                LLMRankVote("h0", rank=2, confidence=0.90),
                LLMRankVote("h2", rank=3, confidence=0.80),
            ),
        )
        self.assertEqual(ranked.observed_candidate_id, "h0")
        self.assertEqual(ranked.observed_transcript, "音声認識です")
        self.assertEqual(ranked.llm_preferred_candidate_id, "h1")
        self.assertLessEqual(
            max(abs(item.llm_tiebreak_score) for item in ranked.hypotheses),
            0.15,
        )

    def test_transcript_state_rejects_observed_text_mutation(self) -> None:
        candidate = self._candidates()[0]
        ranked = RankedHypothesis(
            hypothesis=candidate, acoustic_score=1.0, rank=1
        )
        with self.assertRaisesRegex(ValueError, "selected acoustic candidate"):
            TranscriptState(
                observed_transcript="LLMが書き換えた文",
                observed_candidate_id=candidate.candidate_id,
                hypotheses=(ranked,),
                mora_units=(),
            )

    def test_normalized_transcript_is_separate(self) -> None:
        state = select_observed_transcript(self._candidates())
        normalized = attach_normalized_transcript(
            state,
            "音声認識です。",
            method="deterministic-ja-v1",
        )
        self.assertEqual(normalized.observed_transcript, "音声認識です")
        self.assertEqual(normalized.normalized_transcript, "音声認識です。")


class WhisperNBestTests(unittest.TestCase):
    class _Result:
        sequences_ids = ((10, 11), (20, 21, 22))
        scores = (-0.5, -0.6)
        no_speech_prob = 0.1

    class _RawModel:
        def __init__(self) -> None:
            self.kwargs = None

        def generate(self, encoded, prompts, **kwargs):
            self.kwargs = kwargs
            return [WhisperNBestTests._Result()]

    class _Wrapper:
        def __init__(self) -> None:
            self.model = WhisperNBestTests._RawModel()

    class _Tokenizer:
        non_speech_tokens = (50, 51)
        transcribe = 60
        translate = 61
        sot = 62
        sot_prev = 63
        sot_lm = 64
        no_speech = 65

        @staticmethod
        def decode(ids):
            return "|".join(str(value) for value in ids)

    def test_resolves_faster_whisper_suppression_sentinel(self) -> None:
        resolved = resolve_faster_whisper_suppress_tokens(
            self._Tokenizer(), (-1, 99)
        )
        self.assertEqual(resolved, (50, 51, 60, 61, 62, 63, 64, 65, 99))

    def test_decode_uses_faster_whisper_default_suppression_with_tokenizer(self) -> None:
        wrapper = self._Wrapper()
        items = decode_nbest_window(
            wrapper,
            encoded_features=object(),
            prompt_tokens=(1, 2, 3),
            num_hypotheses=2,
            tokenizer=self._Tokenizer(),
        )
        self.assertEqual(items[0].text, "10|11")
        self.assertEqual(
            wrapper.model.kwargs["suppress_tokens"],
            [50, 51, 60, 61, 62, 63, 64, 65],
        )

    def test_ct2_adapter_returns_sequences_and_scores(self) -> None:
        wrapper = self._Wrapper()
        items = decode_nbest_window(
            wrapper,
            encoded_features=object(),
            prompt_tokens=(1, 2, 3),
            num_hypotheses=2,
            beam_size=1,
            window_id="0042",
            decode_tokens=lambda ids: ":".join(str(value) for value in ids),
        )
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].candidate_id, "w0042-h00")
        self.assertEqual(items[0].text, "10:11")
        self.assertEqual(wrapper.model.kwargs["num_hypotheses"], 2)
        self.assertEqual(wrapper.model.kwargs["beam_size"], 2)
        self.assertAlmostEqual(items[0].average_logprob, -1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
