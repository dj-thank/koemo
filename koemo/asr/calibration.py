"""Post-hoc calibration utilities for N-best ASR scores.

The functions are deliberately dependency-free so calibration can run in Koemo's
small Windows distribution and in CI. Temperature fitting must use held-out
utterances; fitting and reporting on the same samples gives optimistic numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log
from typing import Iterable, Sequence

_EPSILON = 1e-12


def softmax(logits: Sequence[float], *, temperature: float = 1.0) -> tuple[float, ...]:
    """Return a numerically stable softmax distribution."""

    if not logits:
        raise ValueError("at least one logit is required")
    if not isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    values = tuple(float(value) for value in logits)
    if any(not isfinite(value) for value in values):
        raise ValueError("all logits must be finite")

    scaled = tuple(value / temperature for value in values)
    maximum = max(scaled)
    exponentials = tuple(exp(value - maximum) for value in scaled)
    denominator = sum(exponentials)
    if not isfinite(denominator) or denominator <= 0.0:
        raise ValueError("softmax normalization failed")
    return tuple(value / denominator for value in exponentials)


def normalized_entropy(probabilities: Sequence[float]) -> float:
    """Entropy divided by its maximum, in ``[0, 1]`` for valid inputs."""

    values = tuple(float(value) for value in probabilities)
    if not values:
        raise ValueError("at least one probability is required")
    if any(not isfinite(value) or value < 0.0 for value in values):
        raise ValueError("probabilities must be finite and non-negative")
    total = sum(values)
    if total <= 0.0:
        raise ValueError("probability mass must be positive")
    normalized = tuple(value / total for value in values)
    if len(normalized) == 1:
        return 0.0
    entropy = -sum(value * log(max(value, _EPSILON)) for value in normalized)
    return entropy / log(len(normalized))


@dataclass(frozen=True, slots=True)
class CalibrationExample:
    """One held-out N-best list with the index of the correct candidate."""

    logits: tuple[float, ...]
    correct_index: int
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.logits:
            raise ValueError("calibration example requires at least one logit")
        if any(not isfinite(value) for value in self.logits):
            raise ValueError("calibration logits must be finite")
        if not 0 <= self.correct_index < len(self.logits):
            raise ValueError("correct_index is outside the N-best list")
        if not isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("weight must be finite and positive")


@dataclass(frozen=True, slots=True)
class CalibrationMetrics:
    temperature: float
    negative_log_likelihood: float
    brier_score: float
    expected_calibration_error: float
    example_count: int


def evaluate_calibration(
    examples: Iterable[CalibrationExample],
    *,
    temperature: float = 1.0,
    bins: int = 15,
) -> CalibrationMetrics:
    """Compute multiclass NLL, Brier score, and top-label ECE."""

    items = tuple(examples)
    if not items:
        raise ValueError("at least one calibration example is required")
    if bins < 2:
        raise ValueError("bins must be >= 2")

    total_weight = sum(item.weight for item in items)
    nll = 0.0
    brier = 0.0
    bucket_weight = [0.0] * bins
    bucket_confidence = [0.0] * bins
    bucket_correct = [0.0] * bins

    for item in items:
        probabilities = softmax(item.logits, temperature=temperature)
        correct_probability = max(probabilities[item.correct_index], _EPSILON)
        nll += -log(correct_probability) * item.weight
        brier += (
            sum(
                (probability - (1.0 if index == item.correct_index else 0.0)) ** 2
                for index, probability in enumerate(probabilities)
            )
            * item.weight
        )

        predicted_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        confidence = probabilities[predicted_index]
        bucket = min(bins - 1, int(confidence * bins))
        bucket_weight[bucket] += item.weight
        bucket_confidence[bucket] += confidence * item.weight
        bucket_correct[bucket] += (
            1.0 if predicted_index == item.correct_index else 0.0
        ) * item.weight

    ece = 0.0
    for weight, confidence_sum, correct_sum in zip(
        bucket_weight, bucket_confidence, bucket_correct
    ):
        if weight <= 0.0:
            continue
        average_confidence = confidence_sum / weight
        accuracy = correct_sum / weight
        ece += (weight / total_weight) * abs(accuracy - average_confidence)

    return CalibrationMetrics(
        temperature=temperature,
        negative_log_likelihood=nll / total_weight,
        brier_score=brier / total_weight,
        expected_calibration_error=ece,
        example_count=len(items),
    )


def fit_temperature(
    examples: Iterable[CalibrationExample],
    *,
    minimum: float = 0.15,
    maximum: float = 8.0,
    grid_size: int = 81,
    refinement_rounds: int = 4,
) -> CalibrationMetrics:
    """Fit one scalar temperature by deterministic log-space search.

    A grid search is slower than gradient descent but robust for very small
    calibration sets, deterministic across platforms, and has no ML dependency.
    The returned metrics are evaluated at the best temperature.
    """

    items = tuple(examples)
    if not items:
        raise ValueError("at least one calibration example is required")
    if not 0.0 < minimum < maximum:
        raise ValueError("temperature bounds must satisfy 0 < minimum < maximum")
    if grid_size < 3:
        raise ValueError("grid_size must be >= 3")
    if refinement_rounds < 0:
        raise ValueError("refinement_rounds must be non-negative")

    lower_log = log(minimum)
    upper_log = log(maximum)
    best_metrics: CalibrationMetrics | None = None

    for _ in range(refinement_rounds + 1):
        step = (upper_log - lower_log) / (grid_size - 1)
        temperatures = tuple(exp(lower_log + index * step) for index in range(grid_size))
        metrics = tuple(
            evaluate_calibration(items, temperature=temperature)
            for temperature in temperatures
        )
        best_index = min(
            range(len(metrics)),
            key=lambda index: (
                metrics[index].negative_log_likelihood,
                abs(log(metrics[index].temperature)),
            ),
        )
        best_metrics = metrics[best_index]

        left_index = max(0, best_index - 1)
        right_index = min(grid_size - 1, best_index + 1)
        lower_log = log(temperatures[left_index])
        upper_log = log(temperatures[right_index])
        if left_index == right_index:
            break

    assert best_metrics is not None
    return best_metrics
