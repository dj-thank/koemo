from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from moraweave.training import MultiTaskConfig, build_model


class FakeEncoder(nn.Module):
    def forward(self, *, input_features, return_dict=True):
        return SimpleNamespace(last_hidden_state=input_features)


class FakeWhisper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(d_model=8)
        self.model = SimpleNamespace(encoder=FakeEncoder())
        self.text_head = nn.Linear(8, 11)

    def forward(self, *, encoder_outputs, labels, return_dict=True):
        logits = self.text_head(encoder_outputs.last_hidden_state)
        return SimpleNamespace(loss=logits.square().mean(), logits=logits)


def test_all_auxiliary_heads_share_encoder_and_backpropagate() -> None:
    model = build_model(
        FakeWhisper(),
        MultiTaskConfig(mora_vocab_size=16, phone_vocab_size=24),
    )
    features = torch.randn(2, 9, 8)
    result = model(
        input_features=features,
        text_labels=torch.zeros(2, 2, dtype=torch.long),
        mora_labels=torch.tensor([[1, 2, 3, -100], [2, 3, -100, -100]]),
        phone_labels=torch.tensor([[1, 2, 3, 4], [2, 3, 4, -100]]),
        boundary_labels=torch.zeros(2, 9, dtype=torch.long),
        f0_targets=torch.randn(2, 9),
        accent_labels=torch.zeros(2, 9, dtype=torch.long),
        preservation_labels=torch.zeros(2, 9, dtype=torch.long),
    )
    assert result["loss"] is not None
    result["loss"].backward()
    assert model.mora_head.weight.grad is not None
    assert model.phone_head.weight.grad is not None
    assert model.boundary_head.weight.grad is not None
    assert model.f0_head.weight.grad is not None
    assert model.accent_head.weight.grad is not None
    assert model.preservation_head.weight.grad is not None
    assert result["evidence_gate"].shape == (2, 9, 4)


def test_ctc_target_blank_is_rejected() -> None:
    model = build_model(
        FakeWhisper(),
        MultiTaskConfig(mora_vocab_size=8, phone_vocab_size=8, blank_id=0),
    )
    with pytest.raises(ValueError):
        model(
            input_features=torch.randn(1, 6, 8),
            mora_labels=torch.tensor([[1, 0, 2]]),
        )
