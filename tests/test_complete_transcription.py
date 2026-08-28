from __future__ import annotations

import json
import tempfile
import unittest
import urllib.request
from pathlib import Path

from japanese_transcriber.diarization import assign_speakers, parse_rttm, relabel_speakers
from japanese_transcriber.engine import EngineConfig
from japanese_transcriber.formatters import render_srt, render_vtt
from japanese_transcriber.local_llm import (
    NormalizationGuard,
    build_local_opener,
    validate_local_endpoint,
    validate_normalized_block,
)
from japanese_transcriber.mora import split_mora
from japanese_transcriber.normalization import deterministic_normalize, join_fragments
from japanese_transcriber.pipeline import PipelineConfig, transcribe_file, verify_observed_integrity
from japanese_transcriber.types import EngineResult, Segment, Word


class FakeEngine:
    def transcribe(self, audio_path: str | Path) -> EngineResult:
        return EngineResult(
            engine={"name": "fake", "model": "fixture"},
            language={"code": "ja", "probability": 0.99},
            duration={"seconds": 3.0, "secondsAfterVad": 2.5},
            segments=[
                Segment(
                    id="seg-000000",
                    index=0,
                    start=0.0,
                    end=1.4,
                    text=" 今日は、",
                    avg_logprob=-0.2,
                    no_speech_prob=0.01,
                    compression_ratio=1.1,
                    words=[Word(0, "今日", 0.0, 0.6, 0.94), Word(1, "は", 0.6, 0.9, 0.91)],
                ),
                Segment(
                    id="seg-000001",
                    index=1,
                    start=1.5,
                    end=2.9,
                    text="学校へ行きます。 ",
                    avg_logprob=-1.2,
                    no_speech_prob=0.05,
                    compression_ratio=1.0,
                    words=[Word(2, "学校", 1.5, 2.0, 0.84), Word(3, "へ", 2.0, 2.2, 0.42), Word(4, "行きます", 2.2, 2.9, 0.87)],
                ),
            ],
        )


class MoraTests(unittest.TestCase):
    def test_packaged_mora_tokenizer(self) -> None:
        self.assertEqual([unit.kana for unit in split_mora("がっこう")], ["ガ", "ッ", "コ", "ウ"])
        self.assertEqual([unit.kana for unit in split_mora("きゃく")], ["キャ", "ク"])
        self.assertEqual([unit.kana for unit in split_mora("ｽｰﾊﾟｰ")], ["ス", "ー", "パ", "ー"])


class NormalizationTests(unittest.TestCase):
    def test_joins_japanese_without_artificial_space(self) -> None:
        self.assertEqual(join_fragments(["今日は、", " 学校へ行きます。"]), "今日は、学校へ行きます。")
        self.assertEqual(join_fragments(["OpenAI", " API"]), "OpenAI API")

    def test_deterministic_normalization(self) -> None:
        self.assertEqual(deterministic_normalize("ＡＩ  、 便利！！！！！"), "AI、 便利！！")

    def test_local_endpoint_guard(self) -> None:
        self.assertEqual(validate_local_endpoint("http://127.0.0.1:11434"), "http://127.0.0.1:11434/api/chat")
        with self.assertRaises(ValueError):
            validate_local_endpoint("https://example.com/api/chat")
        with self.assertRaises(ValueError):
            validate_local_endpoint("http://127.0.0.1:11434/api/chat?redirect=1")

    def test_local_opener_disables_proxy_and_redirect_transport(self) -> None:
        opener = build_local_opener()
        proxy_handlers = [handler for handler in opener.handlers if isinstance(handler, urllib.request.ProxyHandler)]
        self.assertEqual(len(proxy_handlers), 1)
        self.assertEqual(proxy_handlers[0].proxies, {})
        self.assertTrue(any(type(handler).__name__ == "_NoRedirectHandler" for handler in opener.handlers))

    def test_llm_block_validation_falls_back_on_divergence(self) -> None:
        block = [Segment("seg-000000", 0, 0, 1, "昨日学校を行きました")]
        accepted, rejected = validate_normalized_block(
            block,
            {"segments": [{"id": "seg-000000", "text": "明日は宇宙へ行きます"}]},
            NormalizationGuard(min_similarity=0.8),
        )
        self.assertEqual(rejected, ["seg-000000"])
        self.assertEqual(accepted[0]["text"], "昨日学校を行きました")


class DiarizationTests(unittest.TestCase):
    def test_rttm_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rttm = Path(directory) / "sample.rttm"
            rttm.write_text(
                "SPEAKER file 1 0.000 1.400 <NA> <NA> spk_A <NA> <NA>\n"
                "SPEAKER file 1 1.400 1.600 <NA> <NA> spk_B <NA> <NA>\n",
                encoding="utf-8",
            )
            segments = FakeEngine().transcribe("x").segments
            assign_speakers(segments, parse_rttm(rttm))
            mapping = relabel_speakers(segments)
            self.assertEqual(mapping, {"spk_A": "話者1", "spk_B": "話者2"})
            self.assertEqual([segment.speaker for segment in segments], ["話者1", "話者2"])


class PipelineTests(unittest.TestCase):
    def test_pipeline_writes_outputs_and_preserves_observed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "sample.wav"
            audio.write_bytes(b"RIFF-fixture")
            document, outputs = transcribe_file(
                audio,
                engine=FakeEngine(),
                config=PipelineConfig(
                    output_dir=root / "out",
                    formats={"json", "txt", "observed-txt", "srt", "vtt", "md", "tsv", "words-jsonl"},
                ),
            )
            self.assertEqual(document["observedTranscript"]["text"], "今日は、学校へ行きます。")
            self.assertEqual(document["normalizedTranscript"]["observedSha256"], document["observedTranscript"]["sha256"])
            self.assertIn("seg-000001", document["quality"]["uncertainSegmentIds"])
            self.assertEqual(set(outputs), {"json", "txt", "observed-txt", "srt", "vtt", "md", "tsv", "words-jsonl"})
            for path in outputs.values():
                self.assertTrue(Path(path).is_file())

            payload = json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["schemaVersion"], "1.0.0")
            self.assertEqual(payload["source"]["name"], "sample.wav")
            self.assertEqual(payload["source"]["path"], "sample.wav")
            self.assertTrue(verify_observed_integrity(payload))
            payload["observedTranscript"]["text"] = "改ざん"
            with self.assertRaises(ValueError):
                verify_observed_integrity(payload)
            self.assertIn("00:00:00,000 --> 00:00:01,400", Path(outputs["srt"]).read_text(encoding="utf-8"))
            self.assertTrue(Path(outputs["vtt"]).read_text(encoding="utf-8").startswith("WEBVTT"))

    def test_subtitle_renderers_use_speaker_labels(self) -> None:
        document = {
            "observedTranscript": {"segments": [{"id": "s", "start": 0, "end": 1, "speaker": "話者1", "text": "はい"}]},
            "normalizedTranscript": None,
        }
        self.assertIn("【話者1】はい", render_srt(document, variant="observed"))
        self.assertIn("【話者1】はい", render_vtt(document, variant="observed"))


class EngineConfigTests(unittest.TestCase):
    def test_japanese_accuracy_oriented_defaults(self) -> None:
        config = EngineConfig()
        self.assertEqual(config.language, "ja")
        self.assertTrue(config.word_timestamps)
        self.assertTrue(config.vad_filter)
        self.assertEqual(config.model, "large-v3-turbo")


if __name__ == "__main__":
    unittest.main()
