"""Whisper encoder with text, mora-CTC, optional phone-CTC, and boundary heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(slots=True)
class MoraMultitaskOutput:
    loss: Tensor | None
    text_loss: Tensor | None
    mora_ctc_loss: Tensor | None
    phone_ctc_loss: Tensor | None
    boundary_loss: Tensor | None
    text_logits: Tensor | None
    mora_logits: Tensor
    phone_logits: Tensor | None
    boundary_logits: Tensor
    encoder_hidden_states: Tensor


def _ctc_loss(logits: Tensor, labels: Tensor, blank_id: int = 0) -> Tensor:
    batch, frames, _ = logits.shape
    input_lengths = torch.full(
        (batch,), frames, dtype=torch.long, device=logits.device
    )
    valid = labels.ne(-100)
    target_lengths = valid.sum(dim=1).to(dtype=torch.long)
    targets = labels.masked_select(valid).to(dtype=torch.long)
    log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
    return F.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank_id,
        zero_infinity=True,
    )


class MoraMultitaskWhisper(nn.Module):
    def __init__(
        self,
        whisper: nn.Module,
        *,
        mora_vocab_size: int,
        phone_vocab_size: int | None = None,
        boundary_classes: int = 3,
        mora_blank_id: int = 0,
        phone_blank_id: int = 0,
        text_weight: float = 1.0,
        mora_weight: float = 0.4,
        phone_weight: float = 0.2,
        boundary_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if mora_vocab_size < 2:
            raise ValueError("mora_vocab_size must include blank plus at least one label")

        self.whisper = whisper
        hidden_size = int(whisper.config.d_model)
        self.mora_head = nn.Linear(hidden_size, mora_vocab_size)
        self.phone_head = (
            nn.Linear(hidden_size, phone_vocab_size)
            if phone_vocab_size is not None
            else None
        )
        self.boundary_head = nn.Linear(hidden_size, boundary_classes)
        self.mora_blank_id = mora_blank_id
        self.phone_blank_id = phone_blank_id
        self.loss_weights = {
            "text": text_weight,
            "mora": mora_weight,
            "phone": phone_weight,
            "boundary": boundary_weight,
        }

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        mora_vocab_size: int,
        phone_vocab_size: int | None = None,
        **kwargs: Any,
    ) -> "MoraMultitaskWhisper":
        try:
            from transformers import WhisperForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("install transformers before loading Whisper") from exc
        whisper = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)
        return cls(
            whisper,
            mora_vocab_size=mora_vocab_size,
            phone_vocab_size=phone_vocab_size,
            **kwargs,
        )

    def forward(
        self,
        *,
        input_features: Tensor,
        labels: Tensor | None = None,
        mora_labels: Tensor | None = None,
        phone_labels: Tensor | None = None,
        boundary_labels: Tensor | None = None,
    ) -> MoraMultitaskOutput:
        encoder = self.whisper.model.encoder(input_features=input_features, return_dict=True)
        hidden = encoder.last_hidden_state
        mora_logits = self.mora_head(hidden)
        phone_logits = self.phone_head(hidden) if self.phone_head is not None else None
        boundary_logits = self.boundary_head(hidden)

        text_loss = None
        text_logits = None
        if labels is not None:
            text_output = self.whisper(
                encoder_outputs=encoder,
                labels=labels,
                return_dict=True,
            )
            text_loss = text_output.loss
            text_logits = text_output.logits

        mora_loss = (
            _ctc_loss(mora_logits, mora_labels, self.mora_blank_id)
            if mora_labels is not None
            else None
        )
        phone_loss = (
            _ctc_loss(phone_logits, phone_labels, self.phone_blank_id)
            if phone_logits is not None and phone_labels is not None
            else None
        )
        boundary_loss = None
        if boundary_labels is not None:
            if boundary_labels.shape[:2] != boundary_logits.shape[:2]:
                raise ValueError("boundary labels must match encoder frame length")
            boundary_loss = F.cross_entropy(
                boundary_logits.reshape(-1, boundary_logits.shape[-1]),
                boundary_labels.reshape(-1),
                ignore_index=-100,
            )

        weighted = []
        for name, value in (
            ("text", text_loss),
            ("mora", mora_loss),
            ("phone", phone_loss),
            ("boundary", boundary_loss),
        ):
            if value is not None:
                weighted.append(self.loss_weights[name] * value)
        loss = torch.stack(weighted).sum() if weighted else None

        return MoraMultitaskOutput(
            loss=loss,
            text_loss=text_loss,
            mora_ctc_loss=mora_loss,
            phone_ctc_loss=phone_loss,
            boundary_loss=boundary_loss,
            text_logits=text_logits,
            mora_logits=mora_logits,
            phone_logits=phone_logits,
            boundary_logits=boundary_logits,
            encoder_hidden_states=hidden,
        )

    def save_auxiliary_heads(self, path: str) -> None:
        torch.save(
            {
                "mora_head": self.mora_head.state_dict(),
                "phone_head": None if self.phone_head is None else self.phone_head.state_dict(),
                "boundary_head": self.boundary_head.state_dict(),
                "loss_weights": self.loss_weights,
                "mora_blank_id": self.mora_blank_id,
                "phone_blank_id": self.phone_blank_id,
            },
            path,
        )
