from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class MultiTaskConfig:
    mora_vocab_size: int
    phone_vocab_size: int
    boundary_classes: int = 2
    accent_classes: int = 5
    preservation_classes: int = 4
    dropout: float = 0.1
    text_weight: float = 1.0
    mora_weight: float = 0.45
    phone_weight: float = 0.20
    boundary_weight: float = 0.12
    f0_weight: float = 0.08
    accent_weight: float = 0.08
    preservation_weight: float = 0.18
    blank_id: int = 0


def require_torch() -> tuple[Any, Any, Any]:
    try:
        import torch
        from torch import nn
        from torch.nn import functional as functional
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("install MoraWeave with the train extra") from exc
    return torch, nn, functional


def build_model(base_whisper: Any, config: MultiTaskConfig) -> Any:
    """Build a shared-encoder Whisper extension without replacing its text decoder."""

    torch, nn, functional = require_torch()
    hidden_size = int(base_whisper.config.d_model)

    class FourBranchGate(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.branches = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(hidden_size),
                        nn.Linear(hidden_size, hidden_size),
                        nn.SiLU(),
                        nn.Dropout(config.dropout),
                    )
                    for _ in range(4)
                ]
            )
            self.router = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, 4),
            )
            self.output_norm = nn.LayerNorm(hidden_size)

        def forward(self, hidden: Any) -> tuple[Any, Any]:
            gate = self.router(hidden).softmax(dim=-1)
            stacked = torch.stack([branch(hidden) for branch in self.branches], dim=-2)
            mixed = (stacked * gate.unsqueeze(-1)).sum(dim=-2)
            return self.output_norm(hidden + mixed), gate

    class MoraWeaveWhisper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base_whisper = base_whisper
            self.evidence_gate = FourBranchGate()
            self.mora_head = nn.Linear(hidden_size, config.mora_vocab_size)
            self.phone_head = nn.Linear(hidden_size, config.phone_vocab_size)
            self.boundary_head = nn.Linear(hidden_size, config.boundary_classes)
            self.f0_head = nn.Linear(hidden_size, 1)
            self.accent_head = nn.Linear(hidden_size, config.accent_classes)
            self.preservation_head = nn.Linear(hidden_size, config.preservation_classes)

        @staticmethod
        def _ctc_loss(logits: Any, labels: Any, lengths: Any | None = None) -> Any:
            valid = labels.ne(-100)
            targets = labels.masked_select(valid).to(dtype=torch.long)
            target_lengths = valid.sum(dim=1).to(dtype=torch.long)
            batch, frames, _ = logits.shape
            if lengths is None:
                lengths = torch.full(
                    (batch,), frames, dtype=torch.long, device=logits.device
                )
            if (targets == config.blank_id).any():
                raise ValueError("CTC targets may not contain the blank ID")
            return functional.ctc_loss(
                logits.log_softmax(dim=-1).transpose(0, 1),
                targets,
                lengths.to(dtype=torch.long),
                target_lengths,
                blank=config.blank_id,
                zero_infinity=True,
            )

        @staticmethod
        def _frame_loss(logits: Any, labels: Any) -> Any:
            return functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )

        def forward(
            self,
            *,
            input_features: Any,
            text_labels: Any | None = None,
            mora_labels: Any | None = None,
            phone_labels: Any | None = None,
            encoder_lengths: Any | None = None,
            boundary_labels: Any | None = None,
            f0_targets: Any | None = None,
            f0_mask: Any | None = None,
            accent_labels: Any | None = None,
            preservation_labels: Any | None = None,
        ) -> dict[str, Any]:
            encoder_output = self.base_whisper.model.encoder(
                input_features=input_features, return_dict=True
            )
            hidden, gate = self.evidence_gate(encoder_output.last_hidden_state)

            losses: dict[str, Any] = {}
            text_logits = None
            if text_labels is not None:
                encoder_output.last_hidden_state = hidden
                text_output = self.base_whisper(
                    encoder_outputs=encoder_output,
                    labels=text_labels,
                    return_dict=True,
                )
                losses["text"] = text_output.loss
                text_logits = text_output.logits

            mora_logits = self.mora_head(hidden)
            phone_logits = self.phone_head(hidden)
            boundary_logits = self.boundary_head(hidden)
            f0 = self.f0_head(hidden).squeeze(-1)
            accent_logits = self.accent_head(hidden)
            preservation_logits = self.preservation_head(hidden)

            if mora_labels is not None:
                losses["mora"] = self._ctc_loss(mora_logits, mora_labels, encoder_lengths)
            if phone_labels is not None:
                losses["phone"] = self._ctc_loss(phone_logits, phone_labels, encoder_lengths)
            if boundary_labels is not None:
                losses["boundary"] = self._frame_loss(boundary_logits, boundary_labels)
            if accent_labels is not None:
                losses["accent"] = self._frame_loss(accent_logits, accent_labels)
            if preservation_labels is not None:
                losses["preservation"] = self._frame_loss(
                    preservation_logits, preservation_labels
                )
            if f0_targets is not None:
                mask = (
                    f0_mask.to(dtype=torch.bool)
                    if f0_mask is not None
                    else torch.isfinite(f0_targets)
                )
                if mask.any():
                    losses["f0"] = functional.smooth_l1_loss(f0[mask], f0_targets[mask])

            weights = {
                "text": config.text_weight,
                "mora": config.mora_weight,
                "phone": config.phone_weight,
                "boundary": config.boundary_weight,
                "f0": config.f0_weight,
                "accent": config.accent_weight,
                "preservation": config.preservation_weight,
            }
            total = None
            for name, loss in losses.items():
                weighted = weights[name] * loss
                total = weighted if total is None else total + weighted

            return {
                "loss": total,
                "losses": losses,
                "text_logits": text_logits,
                "mora_logits": mora_logits,
                "phone_logits": phone_logits,
                "boundary_logits": boundary_logits,
                "f0": f0,
                "accent_logits": accent_logits,
                "preservation_logits": preservation_logits,
                "evidence_gate": gate,
                "encoder_hidden_states": hidden,
            }

    return MoraWeaveWhisper()
