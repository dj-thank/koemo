from __future__ import annotations

import unittest

from koemo.asr import (
    CalibrationExample,
    ConsensusConfig,
    DecisionStatus,
    HypothesisFeatures,
    RankedHypothesis,
    TranscriptHypothesis,
    character_error_rate,
    decide_mora_consensus,
    evaluate_calibration,
    fit_temperature,
    learner_error_preservation,
    mora_edit_distance,
    mora_error_rate,
    select_consensus_observed_transcript,
    softmax,
)


class CalibrationTests(unittest.TestCase):
    def test_softmax_is_stable_for_large_logits(self) -> None:
        probabilities = softmax((1000.0, 999.0))
        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertGreater(probabilities[0], probabilities[1])

    def test_temperature_fit_improves_held_out_nll(self) -> None:
        examples = (
            CalibrationExample((8.0, 0.0), correct_index=1),
            CalibrationExample((8.0, 0.0), correct_index=0),
            CalibrationExample((8.0, 0.0), correct_index=0),
            CalibrationExample((8.0, 0.0), correct_index=0),
        )
        baseline = evaluate_calibration(examples)
        fitted = fit_temperature(examples)
        self.assertLessEqual(
            fitted.negative_log_likelihood,
            baseline.negative_log_likelihood,
        )
        self.assertGreater(fitted.temperature, 1.0)


class MoraConsensusTests(unittest.TestCase):
    @staticmethod
    def _ranked(
        candidates: tuple[tuple[str, str, float, float], ...],
    ) -> tuple[RankedHypothesis, ...]:
        return tuple(
            RankedHypothesis(
                hypothesis=TranscriptHypothesis(
                    candidate_id=candidate_id,
                    text=text,
                    features=HypothesisFeatures(
                        whisper_logprob=score,
                        no_speech_probability=no_speech,
                    ),
                ),
                acoustic_score=score,
                rank=rank,
            )
            for rank, (candidate_id, text, score, no_speech) in enumerate(
                candidates, start=1
            )
        )

    def test_duplicate_beams_do_not_multiply_probability_mass(self) -> None:
        ranked = self._ranked(
            (
                ("h0", "カキク", 2.0, 0.01),
                ("h1", "カキケ", 1.9, 0.01),
                ("h2", "カキケ", 1.8, 0.01),
                ("h3", "カキケ", 1.7, 0.01),
            )
        )
        decision = decide_mora_consensus(ranked)
        self.assertEqual(len(decision.candidates), 2)
        duplicate_group = next(
            candidate
            for candidate in decision.candidates
            if candidate.candidate_id == "h1"
        )
        self.assertEqual(
            duplicate_group.duplicate_candidate_ids,
            ("h1", "h2", "h3"),
        )

    def test_mbr_can_override_an_isolated_acoustic_top(self) -> None:
        ranked = self._ranked(
            (
                ("h0", "サシス", 2.0, 0.01),
                ("h1", "カキケ", 1.9, 0.01),
                ("h2", "カキコ", 1.8, 0.01),
                ("h3", "カキカ", 1.7, 0.01),
            )
        )
        decision = decide_mora_consensus(
            ranked,
            config=ConsensusConfig(
                acoustic_tiebreak_weight=0.0,
                min_selected_posterior=0.0,
                min_decision_margin=0.0,
                max_normalized_entropy=1.0,
                max_bayes_risk=1.0,
            ),
        )
        self.assertEqual(decision.acoustic_top_candidate_id, "h0")
        self.assertEqual(decision.selected_candidate_id, "h1")
        self.assertTrue(decision.overrode_acoustic_top)

    def test_selection_freezes_only_an_existing_candidate(self) -> None:
        hypotheses = tuple(
            TranscriptHypothesis(
                candidate_id=candidate_id,
                text=text,
                features=HypothesisFeatures(
                    whisper_logprob=logprob,
                    no_speech_probability=0.01,
                ),
            )
            for candidate_id, text, logprob in (
                ("h0", "サシス", -0.10),
                ("h1", "カキケ", -0.20),
                ("h2", "カキコ", -0.30),
                ("h3", "カキカ", -0.40),
            )
        )
        state, decision = select_consensus_observed_transcript(
            hypotheses,
            config=ConsensusConfig(
                posterior_temperature=3.0,
                acoustic_tiebreak_weight=0.0,
                min_selected_posterior=0.0,
                min_decision_margin=0.0,
                max_normalized_entropy=1.0,
                max_bayes_risk=1.0,
            ),
        )
        self.assertEqual(decision.selected_candidate_id, "h1")
        self.assertEqual(state.observed_candidate_id, "h1")
        self.assertEqual(state.observed_transcript, "カキケ")
        self.assertIn(state.observed_transcript, {item.text for item in hypotheses})

    def test_reading_resolver_groups_kanji_and_kana_equivalents(self) -> None:
        ranked = self._ranked(
            (
                ("h0", "今日", 1.0, 0.01),
                ("h1", "きょう", 0.9, 0.01),
            )
        )
        readings = {"今日": "キョウ", "きょう": "キョウ"}
        decision = decide_mora_consensus(
            ranked,
            reading_resolver=readings.__getitem__,
        )
        self.assertEqual(len(decision.candidates), 1)
        self.assertEqual(decision.selected_candidate_id, "h0")
        self.assertEqual(
            decision.candidates[0].duplicate_candidate_ids,
            ("h0", "h1"),
        )

    def test_high_no_speech_probability_abstains(self) -> None:
        ranked = self._ranked((("h0", "カ", 1.0, 0.90),))
        decision = decide_mora_consensus(ranked)
        self.assertEqual(decision.status, DecisionStatus.NO_SPEECH)
        self.assertIn("high_no_speech_probability", decision.reasons)

    def test_mora_edit_distance_counts_sokuon_deletion(self) -> None:
        self.assertEqual(
            mora_edit_distance(("キャ", "ッ", "ト"), ("キャ", "ト")),
            1.0,
        )


class JapaneseEvaluationTests(unittest.TestCase):
    def test_character_error_rate_reports_substitution(self) -> None:
        result = character_error_rate("音声認識", "音声人識")
        self.assertEqual(result.substitutions, 1)
        self.assertEqual(result.reference_units, 4)
        self.assertEqual(result.rate, 0.25)

    def test_mora_error_rate_handles_small_kana_and_sokuon(self) -> None:
        result = mora_error_rate("キャット", "キャト")
        self.assertEqual(result.deletions, 1)
        self.assertAlmostEqual(result.rate, 1.0 / 3.0)

    def test_learner_error_preservation_penalizes_normalization(self) -> None:
        preserved = learner_error_preservation("ガッコウ", "ガコウ", "ガコウ")
        normalized = learner_error_preservation("ガッコウ", "ガコウ", "ガッコウ")
        self.assertTrue(preserved.exact_observation)
        self.assertEqual(preserved.preservation_score, 1.0)
        self.assertFalse(preserved.normalized_to_target)
        self.assertTrue(normalized.normalized_to_target)
        self.assertEqual(normalized.preservation_score, 0.0)
        self.assertGreater(
            preserved.preservation_margin,
            normalized.preservation_margin,
        )


if __name__ == "__main__":
    unittest.main()
