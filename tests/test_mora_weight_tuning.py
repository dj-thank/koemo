from __future__ import annotations

import unittest

from koemo.asr import HypothesisFeatures, TranscriptHypothesis
from koemo.asr.weight_tuning import (
    RankingTrainingExample,
    evaluate_score_weights,
    fit_score_weights,
)


class ScoreWeightTuningTests(unittest.TestCase):
    @staticmethod
    def _example(prefix: str) -> RankingTrainingExample:
        return RankingTrainingExample(
            hypotheses=(
                TranscriptHypothesis(
                    candidate_id=f"{prefix}-wrong",
                    text="サシス",
                    features=HypothesisFeatures(
                        whisper_logprob=0.0,
                        mora_ctc_logprob=-4.0,
                    ),
                ),
                TranscriptHypothesis(
                    candidate_id=f"{prefix}-correct",
                    text="カキク",
                    features=HypothesisFeatures(
                        whisper_logprob=-1.0,
                        mora_ctc_logprob=0.0,
                    ),
                ),
            ),
            correct_candidate_id=f"{prefix}-correct",
        )

    def test_fit_learns_to_trust_mora_evidence(self) -> None:
        training = tuple(self._example(f"train-{index}") for index in range(8))
        validation = tuple(self._example(f"valid-{index}") for index in range(2))
        initial_metrics = evaluate_score_weights(validation)
        fitted = fit_score_weights(
            training,
            validation_examples=validation,
            iterations=400,
            patience=100,
            learning_rate=0.10,
            l2_to_initial=0.001,
        )
        self.assertEqual(initial_metrics.top1_accuracy, 0.0)
        self.assertEqual(fitted.fitted_validation_metrics.top1_accuracy, 1.0)
        self.assertLess(
            fitted.fitted_validation_metrics.negative_log_likelihood,
            fitted.initial_validation_metrics.negative_log_likelihood,
        )
        self.assertGreater(fitted.weights.mora_ctc, fitted.weights.whisper)
        self.assertGreater(fitted.best_iteration, 0)

    def test_non_learnable_components_remain_fixed(self) -> None:
        training = tuple(self._example(f"train-{index}") for index in range(4))
        fitted = fit_score_weights(
            training,
            learnable_components=("moraCtc",),
            iterations=200,
            patience=60,
            learning_rate=0.10,
            l2_to_initial=0.0,
        )
        self.assertEqual(fitted.weights.whisper, 1.0)
        self.assertEqual(fitted.weights.char_ctc, 0.45)
        self.assertGreater(fitted.weights.mora_ctc, 0.70)

    def test_rejects_unknown_learnable_component(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown learnable"):
            fit_score_weights(
                (self._example("train"),),
                learnable_components=("notAComponent",),
            )


if __name__ == "__main__":
    unittest.main()
