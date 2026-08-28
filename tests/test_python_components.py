from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.japanese_mora import merge_character_alignment, split_mora, to_katakana
from scripts.whisper_nbest import decode_generation_result


class JapaneseMoraTests(unittest.TestCase):
    def test_normalization(self) -> None:
        self.assertEqual(to_katakana("きゃ ﾃｨ"), "キャ ティ")

    def test_contracted_sound(self) -> None:
        self.assertEqual([unit.kana for unit in split_mora("きゃく")], ["キャ", "ク"])

    def test_special_morae(self) -> None:
        units = split_mora("スーパーとしんぶん")
        self.assertEqual([unit.kana for unit in units[:4]], ["ス", "ー", "パ", "ー"])
        self.assertEqual(sum(unit.type == "moraic-nasal" for unit in units), 2)

    def test_character_alignment_merge(self) -> None:
        units = merge_character_alignment(
            [
                {"char": "き", "startMs": 10, "endMs": 20, "confidence": 0.9},
                {"char": "ゃ", "startMs": 20, "endMs": 30, "confidence": 0.8},
            ]
        )
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].kana, "キャ")
        self.assertEqual((units[0].start_ms, units[0].end_ms), (10.0, 30.0))
        self.assertEqual(units[0].confidence, 0.8)


class FakeTokenizer:
    timestamp_begin = 100

    @staticmethod
    def decode(token_ids: list[int]) -> str:
        return " ".join(map(str, token_ids))


class WhisperNBestTests(unittest.TestCase):
    def test_decodes_multiple_hypotheses(self) -> None:
        result = SimpleNamespace(
            sequences_ids=[[1, 2, 100], [3, 4, 101]],
            scores=[-0.1, -0.2],
        )
        candidates = decode_generation_result(result, FakeTokenizer())
        self.assertEqual([candidate.text for candidate in candidates], ["1 2", "3 4"])
        self.assertEqual([candidate.whisper_score for candidate in candidates], [-0.1, -0.2])

    def test_rejects_score_count_mismatch(self) -> None:
        result = SimpleNamespace(sequences_ids=[[1], [2]], scores=[-0.1])
        with self.assertRaises(RuntimeError):
            decode_generation_result(result, FakeTokenizer())


try:
    import torch
    from torch import nn

    from training.mora_multitask import MoraMultitaskWhisper
except ImportError:  # pragma: no cover - optional local training dependency
    torch = None
    nn = None
    MoraMultitaskWhisper = None


@unittest.skipIf(torch is None, "torch is not installed")
class MultitaskModelTests(unittest.TestCase):
    def test_auxiliary_heads_backpropagate(self) -> None:
        class FakeEncoder(nn.Module):
            def forward(self, input_features, return_dict=True):
                return SimpleNamespace(last_hidden_state=input_features)

        class FakeWhisper(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(d_model=4)
                self.model = SimpleNamespace(encoder=FakeEncoder())
                self.text_head = nn.Linear(4, 7)

            def forward(self, encoder_outputs, labels, return_dict=True):
                logits = self.text_head(encoder_outputs.last_hidden_state)
                return SimpleNamespace(loss=logits.square().mean(), logits=logits)

        model = MoraMultitaskWhisper(
            FakeWhisper(), mora_vocab_size=6, phone_vocab_size=8
        )
        features = torch.randn(2, 6, 4)
        output = model(
            input_features=features,
            labels=torch.zeros(2, 2, dtype=torch.long),
            mora_labels=torch.tensor([[1, 2, 3], [2, 3, -100]]),
            phone_labels=torch.tensor([[1, 2, 3, 4], [2, 3, -100, -100]]),
            boundary_labels=torch.zeros(2, 6, dtype=torch.long),
        )
        self.assertIsNotNone(output.loss)
        output.loss.backward()
        self.assertIsNotNone(model.mora_head.weight.grad)
        self.assertIsNotNone(model.boundary_head.weight.grad)


if __name__ == "__main__":
    unittest.main()
