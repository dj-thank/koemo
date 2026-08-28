from __future__ import annotations

import unittest

from koemo.asr.runtime_compat import (
    assert_faster_whisper_runtime_compatible,
    probe_faster_whisper_runtime,
)


class _RawWhisper:
    is_multilingual = True

    def generate(self):
        return None

    def detect_language(self):
        return None


class _CompatibleModel:
    def __init__(self) -> None:
        self.model = _RawWhisper()
        self.hf_tokenizer = object()
        self.max_length = 448

    def encode(self):
        return None

    def get_prompt(self):
        return None

    def feature_extractor(self):
        return None

    def _split_segments_by_timestamps(self):
        return None

    def add_word_timestamps(self):
        return None


class RuntimeCompatibilityTests(unittest.TestCase):
    def test_accepts_tested_runtime_shape_and_versions(self) -> None:
        report = assert_faster_whisper_runtime_compatible(
            _CompatibleModel(),
            faster_whisper_version="1.2.1",
            ctranslate2_version="4.6.0",
        )
        self.assertTrue(report.compatible)
        self.assertEqual(report.errors, ())
        self.assertIn("model.generate", report.capabilities)

    def test_fails_closed_when_generate_is_missing(self) -> None:
        model = _CompatibleModel()
        model.model.generate = None
        report = probe_faster_whisper_runtime(
            model,
            faster_whisper_version="1.2.1",
            ctranslate2_version="4.6.0",
        )
        self.assertFalse(report.compatible)
        self.assertIn("missing_callable:model.generate", report.errors)

    def test_rejects_unmatched_ctranslate2_major(self) -> None:
        report = probe_faster_whisper_runtime(
            _CompatibleModel(),
            faster_whisper_version="1.2.1",
            ctranslate2_version="5.0.0",
        )
        self.assertFalse(report.compatible)
        self.assertTrue(
            any(value.startswith("unsupported_ctranslate2_version") for value in report.errors)
        )

    def test_newer_faster_whisper_is_structurally_allowed_but_warned(self) -> None:
        report = probe_faster_whisper_runtime(
            _CompatibleModel(),
            faster_whisper_version="1.3.0",
            ctranslate2_version="4.6.0",
        )
        self.assertTrue(report.compatible)
        self.assertTrue(
            any(value.startswith("untested_faster_whisper_version") for value in report.warnings)
        )

    def test_prerelease_suffix_digits_are_not_merged_into_patch_version(self) -> None:
        report = probe_faster_whisper_runtime(
            _CompatibleModel(),
            faster_whisper_version="1.2.1rc1",
            ctranslate2_version="4.6.0rc2",
        )
        self.assertTrue(report.compatible)
        self.assertFalse(
            any(value.startswith("unsupported_ctranslate2_version") for value in report.errors)
        )
        self.assertTrue(
            any(value.startswith("untested_faster_whisper_version") for value in report.warnings)
        )


if __name__ == "__main__":
    unittest.main()
