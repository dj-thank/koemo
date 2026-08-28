from __future__ import annotations

import unittest

from koemo.asr.benchmark import (
    AccuracyGateConfig,
    BenchmarkSplit,
    BenchmarkUtterance,
    NBestCandidate,
    SystemPrediction,
    aggregate_system_metrics,
    evaluate_accuracy_gate,
    evaluate_system,
    evaluate_utterance,
    paired_speaker_bootstrap,
    validate_manifest,
)


class ManifestIntegrityTests(unittest.TestCase):
    def test_detects_speaker_and_audio_leakage(self) -> None:
        digest = "a" * 64
        manifest = (
            BenchmarkUtterance(
                "u1",
                "s1",
                BenchmarkSplit.CALIBRATION,
                "ア",
                audio_sha256=digest,
            ),
            BenchmarkUtterance(
                "u2",
                "s1",
                BenchmarkSplit.TEST,
                "イ",
                audio_sha256=digest,
            ),
        )
        codes = {issue.code for issue in validate_manifest(manifest)}
        self.assertIn("speaker_split_leakage", codes)
        self.assertIn("audio_split_leakage", codes)

    def test_evaluate_system_rejects_missing_predictions(self) -> None:
        manifest = (
            BenchmarkUtterance("u1", "s1", BenchmarkSplit.TEST, "ア"),
            BenchmarkUtterance("u2", "s2", BenchmarkSplit.TEST, "イ"),
        )
        predictions = (SystemPrediction("u1", "candidate", "ア"),)
        with self.assertRaisesRegex(ValueError, "missing predictions"):
            evaluate_system(manifest, predictions)


class OracleAndAggregationTests(unittest.TestCase):
    def test_nbest_oracle_exposes_reranking_headroom(self) -> None:
        utterance = BenchmarkUtterance(
            "u1",
            "s1",
            BenchmarkSplit.TEST,
            "音声認識",
            reference_reading="オンセイニンシキ",
        )
        prediction = SystemPrediction(
            "u1",
            "candidate",
            "音声人識",
            reading="オンセイジンシキ",
            candidates=(
                NBestCandidate(
                    "h0",
                    "音声人識",
                    reading="オンセイジンシキ",
                    score=-0.1,
                ),
                NBestCandidate(
                    "h1",
                    "音声認識",
                    reading="オンセイニンシキ",
                    score=-0.2,
                ),
            ),
        )
        result = evaluate_utterance(utterance, prediction)
        self.assertEqual(result.cer.errors, 1)
        self.assertEqual(result.oracle_cer.errors, 0)
        self.assertEqual(result.oracle_cer_candidate_id, "h1")
        self.assertEqual(result.oracle_mora_error_rate.errors, 0)
        self.assertEqual(result.oracle_mora_candidate_id, "h1")

    def test_aggregate_uses_micro_error_rate(self) -> None:
        manifest = (
            BenchmarkUtterance("u1", "s1", BenchmarkSplit.TEST, "ア"),
            BenchmarkUtterance("u2", "s2", BenchmarkSplit.TEST, "アイウエオ"),
        )
        predictions = (
            SystemPrediction("u1", "candidate", ""),
            SystemPrediction("u2", "candidate", "アイウエオ"),
        )
        metrics = aggregate_system_metrics(evaluate_system(manifest, predictions))
        self.assertEqual(metrics.cer.errors, 1)
        self.assertEqual(metrics.cer.reference_units, 6)
        self.assertAlmostEqual(metrics.cer.rate, 1.0 / 6.0)

    def test_aggregates_learner_fidelity_status_and_latency(self) -> None:
        manifest = (
            BenchmarkUtterance(
                "u1",
                "s1",
                BenchmarkSplit.TEST,
                "学校",
                reference_reading="ガコウ",
                target_reading="ガッコウ",
                observed_reading="ガコウ",
            ),
            BenchmarkUtterance(
                "u2",
                "s2",
                BenchmarkSplit.TEST,
                "猫",
                reference_reading="ネコ",
            ),
        )
        predictions = (
            SystemPrediction(
                "u1",
                "candidate",
                "学校",
                reading="ガコウ",
                status="review",
                latency_ms=100.0,
            ),
            SystemPrediction(
                "u2",
                "candidate",
                "猫",
                reading="ネコ",
                latency_ms=200.0,
            ),
        )
        metrics = aggregate_system_metrics(evaluate_system(manifest, predictions))
        self.assertEqual(metrics.learner_evaluated, 1)
        self.assertEqual(metrics.mean_learner_preservation, 1.0)
        self.assertEqual(metrics.normalized_to_target_rate, 0.0)
        self.assertEqual(metrics.accept_count, 1)
        self.assertEqual(metrics.review_count, 1)
        self.assertEqual(metrics.latency_p50_ms, 150.0)
        self.assertEqual(metrics.latency_p95_ms, 195.0)


class StatisticalGateTests(unittest.TestCase):
    @staticmethod
    def _systems():
        manifest = (
            BenchmarkUtterance("u1", "s1", BenchmarkSplit.TEST, "アイ"),
            BenchmarkUtterance("u2", "s1", BenchmarkSplit.TEST, "カキ"),
            BenchmarkUtterance("u3", "s2", BenchmarkSplit.TEST, "サシ"),
            BenchmarkUtterance("u4", "s2", BenchmarkSplit.TEST, "タチ"),
        )
        baseline_predictions = (
            SystemPrediction("u1", "baseline", "ア"),
            SystemPrediction("u2", "baseline", "カ"),
            SystemPrediction("u3", "baseline", "サ"),
            SystemPrediction("u4", "baseline", "タ"),
        )
        candidate_predictions = tuple(
            SystemPrediction(item.utterance_id, "candidate", item.reference_text)
            for item in manifest
        )
        return (
            evaluate_system(manifest, baseline_predictions),
            evaluate_system(manifest, candidate_predictions),
        )

    def test_paired_bootstrap_resamples_speakers(self) -> None:
        baseline, candidate = self._systems()
        comparison = paired_speaker_bootstrap(
            baseline,
            candidate,
            samples=200,
            seed=7,
        )
        self.assertEqual(comparison.cluster_count, 2)
        self.assertLess(comparison.delta, 0.0)
        self.assertLess(comparison.upper_bound, 0.0)
        self.assertEqual(comparison.improvement_probability, 1.0)

    def test_accuracy_gate_accepts_supported_improvement(self) -> None:
        baseline, candidate = self._systems()
        baseline_metrics = aggregate_system_metrics(baseline)
        candidate_metrics = aggregate_system_metrics(candidate)
        comparison = paired_speaker_bootstrap(
            baseline,
            candidate,
            samples=200,
            seed=7,
        )
        gate = evaluate_accuracy_gate(
            baseline_metrics,
            candidate_metrics,
            config=AccuracyGateConfig(
                require_cer_confidence_improvement=True,
            ),
            cer_bootstrap=comparison,
        )
        self.assertTrue(gate.passed)
        self.assertEqual(gate.reasons, ())

    def test_accuracy_gate_blocks_regression(self) -> None:
        baseline, candidate = self._systems()
        gate = evaluate_accuracy_gate(
            aggregate_system_metrics(candidate),
            aggregate_system_metrics(baseline),
        )
        self.assertFalse(gate.passed)
        self.assertIn("cer_regression", gate.reasons)


if __name__ == "__main__":
    unittest.main()
