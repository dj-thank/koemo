from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

_EPSILON = 1e-7


def _clip(value: float) -> float:
    return min(1.0 - _EPSILON, max(_EPSILON, float(value)))


def _logit(probability: float) -> float:
    value = _clip(probability)
    return math.log(value / (1.0 - value))


def _sigmoid(value: float) -> float:
    if value >= 0:
        decay = math.exp(-value)
        return 1.0 / (1.0 + decay)
    growth = math.exp(value)
    return growth / (1.0 + growth)


@dataclass(frozen=True, slots=True)
class CalibrationProfile:
    name: str
    temperature: float = 1.0
    input_kind: str = "probability"
    center: float = 0.0
    scale: float = 1.0
    direction: float = 1.0
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("calibration profile name is required")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be finite and positive")
        if self.input_kind not in {"probability", "logit", "score"}:
            raise ValueError("unsupported calibration input kind")

    @property
    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def transform(self, value: float | None) -> float | None:
        if value is None:
            return None
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("calibration input must be finite")
        if self.input_kind == "probability":
            raw_logit = _logit(numeric)
        elif self.input_kind == "logit":
            raw_logit = numeric
        else:
            raw_logit = self.direction * (numeric - self.center) / self.scale
        return _sigmoid(raw_logit / self.temperature)


def negative_log_likelihood(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    if len(probabilities) != len(labels) or not labels:
        raise ValueError("probabilities and labels must be equal non-empty sequences")
    losses = [
        -(int(label) * math.log(_clip(probability)) + (1 - int(label)) * math.log(1 - _clip(probability)))
        for probability, label in zip(probabilities, labels, strict=True)
    ]
    return sum(losses) / len(losses)


def brier_score(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    if len(probabilities) != len(labels) or not labels:
        raise ValueError("probabilities and labels must be equal non-empty sequences")
    return sum((float(probability) - int(label)) ** 2 for probability, label in zip(probabilities, labels, strict=True)) / len(labels)


def expected_calibration_error(probabilities: Sequence[float], labels: Sequence[int], *, bins: int = 15) -> float:
    if bins < 2:
        raise ValueError("bins must be at least two")
    if len(probabilities) != len(labels) or not labels:
        raise ValueError("probabilities and labels must be equal non-empty sequences")
    total = len(labels)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            (float(probability), int(label))
            for probability, label in zip(probabilities, labels, strict=True)
            if lower <= float(probability) < upper or (index == bins - 1 and float(probability) == 1.0)
        ]
        if not members:
            continue
        confidence = sum(row[0] for row in members) / len(members)
        accuracy = sum(row[1] for row in members) / len(members)
        error += len(members) / total * abs(accuracy - confidence)
    return error


def fit_temperature(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    if len(probabilities) != len(labels) or len(labels) < 2:
        raise ValueError("temperature fitting requires paired examples")
    logits = [_logit(value) for value in probabilities]
    best_temperature = 1.0
    best_loss = math.inf
    # Deterministic log-space search avoids an optimiser dependency.
    for step in range(801):
        exponent = -3.0 + step * (6.0 / 800.0)
        temperature = math.exp(exponent)
        calibrated = [_sigmoid(logit / temperature) for logit in logits]
        loss = negative_log_likelihood(calibrated, labels)
        if loss < best_loss:
            best_loss = loss
            best_temperature = temperature
    return best_temperature


def risk_coverage_curve(probabilities: Sequence[float], labels: Sequence[int]) -> list[tuple[float, float]]:
    if len(probabilities) != len(labels) or not labels:
        raise ValueError("probabilities and labels must be equal non-empty sequences")
    ordered = sorted(
        zip(probabilities, labels, strict=True),
        key=lambda row: (-float(row[0]), -int(row[1])),
    )
    errors = 0
    curve: list[tuple[float, float]] = []
    for index, (_, label) in enumerate(ordered, 1):
        errors += 1 - int(label)
        curve.append((index / len(ordered), errors / index))
    return curve


def area_under_risk_coverage(curve: Iterable[tuple[float, float]]) -> float:
    points = list(curve)
    if not points:
        raise ValueError("risk-coverage curve must not be empty")
    area = 0.0
    previous_coverage = 0.0
    previous_risk = points[0][1]
    for coverage, risk in points:
        area += (coverage - previous_coverage) * (previous_risk + risk) / 2.0
        previous_coverage = coverage
        previous_risk = risk
    return area
