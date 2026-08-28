"""Dependency-free fitting of Koemo's log-linear N-best fusion weights.

The tuner learns on speaker-disjoint training/calibration examples and can use a
separate validation set for early stopping. Feature standardization remains
identical to runtime ranking by reusing the scoring module's component builder.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log, sqrt
from typing import Iterable, Sequence

from .calibration import softmax
from .schema import TranscriptHypothesis
from .scoring import ScoreWeights, _component_vectors

_COMPONENTS = (
    "whisper",
    "charCtc",
    "moraCtc",
    "alignment",
    "prosody",
    "coverage",
    "noSpeech",
    "compression",
)
_COMPONENT_FIELDS = {
    "whisper": "whisper",
    "charCtc": "char_ctc",
    "moraCtc": "mora_ctc",
    "alignment": "alignment",
    "prosody": "prosody",
    "coverage": "coverage",
    "noSpeech": "no_speech_penalty",
    "compression": "compression_penalty",
}
_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class RankingTrainingExample:
    hypotheses: tuple[TranscriptHypothesis, ...]
    correct_candidate_id: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.hypotheses:
            raise ValueError("ranking example requires at least one hypothesis")
        ids = [item.candidate_id for item in self.hypotheses]
        if len(ids) != len(set(ids)):
            raise ValueError("ranking example candidate IDs must be unique")
        if self.correct_candidate_id not in ids:
            raise ValueError("correct_candidate_id is not present in hypotheses")
        if not isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("example weight must be finite and positive")


@dataclass(frozen=True, slots=True)
class RankingMetrics:
    negative_log_likelihood: float
    top1_accuracy: float
    mean_reciprocal_rank: float
    example_count: int

    def to_dict(self) -> dict[str, int | float]:
        return {
            "negativeLogLikelihood": self.negative_log_likelihood,
            "top1Accuracy": self.top1_accuracy,
            "meanReciprocalRank": self.mean_reciprocal_rank,
            "exampleCount": self.example_count,
        }


@dataclass(frozen=True, slots=True)
class WeightFitResult:
    weights: ScoreWeights
    initial_training_metrics: RankingMetrics
    fitted_training_metrics: RankingMetrics
    initial_validation_metrics: RankingMetrics
    fitted_validation_metrics: RankingMetrics
    iterations_run: int
    best_iteration: int
    learnable_components: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "weights": {
                "whisper": self.weights.whisper,
                "charCtc": self.weights.char_ctc,
                "moraCtc": self.weights.mora_ctc,
                "alignment": self.weights.alignment,
                "prosody": self.weights.prosody,
                "coverage": self.weights.coverage,
                "noSpeechPenalty": self.weights.no_speech_penalty,
                "compressionPenalty": self.weights.compression_penalty,
                "compressionThreshold": self.weights.compression_threshold,
            },
            "initialTrainingMetrics": self.initial_training_metrics.to_dict(),
            "fittedTrainingMetrics": self.fitted_training_metrics.to_dict(),
            "initialValidationMetrics": self.initial_validation_metrics.to_dict(),
            "fittedValidationMetrics": self.fitted_validation_metrics.to_dict(),
            "iterationsRun": self.iterations_run,
            "bestIteration": self.best_iteration,
            "learnableComponents": list(self.learnable_components),
        }


def _weight_vector(weights: ScoreWeights) -> list[float]:
    return [float(getattr(weights, _COMPONENT_FIELDS[name])) for name in _COMPONENTS]


def _score_weights(values: Sequence[float], template: ScoreWeights) -> ScoreWeights:
    by_name = dict(zip(_COMPONENTS, values))
    return ScoreWeights(
        whisper=by_name["whisper"],
        char_ctc=by_name["charCtc"],
        mora_ctc=by_name["moraCtc"],
        alignment=by_name["alignment"],
        prosody=by_name["prosody"],
        coverage=by_name["coverage"],
        no_speech_penalty=by_name["noSpeech"],
        compression_penalty=by_name["compression"],
        compression_threshold=template.compression_threshold,
    )


def _example_matrix(
    example: RankingTrainingExample,
    *,
    template: ScoreWeights,
    feature_clip: float,
) -> tuple[tuple[tuple[float, ...], ...], int]:
    components = _component_vectors(example.hypotheses, template)
    matrix = tuple(
        tuple(
            max(-feature_clip, min(feature_clip, components[name][candidate_index]))
            for name in _COMPONENTS
        )
        for candidate_index in range(len(example.hypotheses))
    )
    correct_index = next(
        index
        for index, hypothesis in enumerate(example.hypotheses)
        if hypothesis.candidate_id == example.correct_candidate_id
    )
    return matrix, correct_index


def _logits(matrix: Sequence[Sequence[float]], weights: Sequence[float]) -> tuple[float, ...]:
    return tuple(
        sum(feature * weight for feature, weight in zip(row, weights))
        for row in matrix
    )


def evaluate_score_weights(
    examples: Iterable[RankingTrainingExample],
    *,
    weights: ScoreWeights = ScoreWeights(),
    feature_clip: float = 8.0,
) -> RankingMetrics:
    items = tuple(examples)
    if not items:
        raise ValueError("at least one ranking example is required")
    if not isfinite(feature_clip) or feature_clip <= 0.0:
        raise ValueError("feature_clip must be finite and positive")

    vector = _weight_vector(weights)
    total_weight = sum(item.weight for item in items)
    nll = 0.0
    correct_weight = 0.0
    reciprocal_rank = 0.0
    for item in items:
        matrix, correct_index = _example_matrix(
            item,
            template=weights,
            feature_clip=feature_clip,
        )
        logits = _logits(matrix, vector)
        probabilities = softmax(logits)
        nll += -log(max(probabilities[correct_index], _EPSILON)) * item.weight
        ranking = sorted(
            range(len(logits)),
            key=lambda index: (-logits[index], item.hypotheses[index].candidate_id),
        )
        rank = ranking.index(correct_index) + 1
        if rank == 1:
            correct_weight += item.weight
        reciprocal_rank += (1.0 / rank) * item.weight

    return RankingMetrics(
        negative_log_likelihood=nll / total_weight,
        top1_accuracy=correct_weight / total_weight,
        mean_reciprocal_rank=reciprocal_rank / total_weight,
        example_count=len(items),
    )


def fit_score_weights(
    training_examples: Iterable[RankingTrainingExample],
    *,
    validation_examples: Iterable[RankingTrainingExample] | None = None,
    initial: ScoreWeights = ScoreWeights(),
    learnable_components: Sequence[str] = _COMPONENTS,
    iterations: int = 600,
    learning_rate: float = 0.08,
    l2_to_initial: float = 0.01,
    maximum_weight: float = 5.0,
    feature_clip: float = 8.0,
    patience: int = 80,
    minimum_improvement: float = 1e-7,
) -> WeightFitResult:
    """Fit non-negative fusion weights with projected gradient descent."""

    training = tuple(training_examples)
    validation = training if validation_examples is None else tuple(validation_examples)
    if not training or not validation:
        raise ValueError("training and validation examples must be non-empty")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    for value, name in (
        (learning_rate, "learning_rate"),
        (maximum_weight, "maximum_weight"),
        (feature_clip, "feature_clip"),
    ):
        if not isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not isfinite(l2_to_initial) or l2_to_initial < 0.0:
        raise ValueError("l2_to_initial must be finite and non-negative")
    if patience < 1:
        raise ValueError("patience must be >= 1")
    if not isfinite(minimum_improvement) or minimum_improvement < 0.0:
        raise ValueError("minimum_improvement must be finite and non-negative")

    learnable = tuple(dict.fromkeys(learnable_components))
    unknown = set(learnable) - set(_COMPONENTS)
    if unknown:
        raise ValueError(f"unknown learnable components: {sorted(unknown)}")
    learnable_indices = {_COMPONENTS.index(name) for name in learnable}

    initial_vector = _weight_vector(initial)
    current = initial_vector.copy()
    training_matrices = tuple(
        (*_example_matrix(item, template=initial, feature_clip=feature_clip), item.weight)
        for item in training
    )

    initial_training = evaluate_score_weights(
        training,
        weights=initial,
        feature_clip=feature_clip,
    )
    initial_validation = evaluate_score_weights(
        validation,
        weights=initial,
        feature_clip=feature_clip,
    )
    best = current.copy()
    best_validation = initial_validation
    best_iteration = 0
    stale = 0
    total_training_weight = sum(item.weight for item in training)
    iterations_run = 0

    for iteration in range(1, iterations + 1):
        gradient = [0.0] * len(_COMPONENTS)
        for matrix, correct_index, example_weight in training_matrices:
            logits = _logits(matrix, current)
            probabilities = softmax(logits)
            for candidate_index, row in enumerate(matrix):
                residual = probabilities[candidate_index] - (
                    1.0 if candidate_index == correct_index else 0.0
                )
                for component_index, feature in enumerate(row):
                    gradient[component_index] += (
                        residual * feature * example_weight
                    )

        step_size = learning_rate / sqrt(1.0 + iteration / 100.0)
        for index in range(len(current)):
            if index not in learnable_indices:
                continue
            gradient_value = gradient[index] / total_training_weight
            gradient_value += l2_to_initial * (current[index] - initial_vector[index])
            current[index] = max(
                0.0,
                min(maximum_weight, current[index] - step_size * gradient_value),
            )

        current_weights = _score_weights(current, initial)
        validation_metrics = evaluate_score_weights(
            validation,
            weights=current_weights,
            feature_clip=feature_clip,
        )
        iterations_run = iteration
        if (
            validation_metrics.negative_log_likelihood
            < best_validation.negative_log_likelihood - minimum_improvement
        ):
            best = current.copy()
            best_validation = validation_metrics
            best_iteration = iteration
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    fitted_weights = _score_weights(best, initial)
    return WeightFitResult(
        weights=fitted_weights,
        initial_training_metrics=initial_training,
        fitted_training_metrics=evaluate_score_weights(
            training,
            weights=fitted_weights,
            feature_clip=feature_clip,
        ),
        initial_validation_metrics=initial_validation,
        fitted_validation_metrics=evaluate_score_weights(
            validation,
            weights=fitted_weights,
            feature_clip=feature_clip,
        ),
        iterations_run=iterations_run,
        best_iteration=best_iteration,
        learnable_components=learnable,
    )
