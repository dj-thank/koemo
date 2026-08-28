"""Compare two Japanese ASR systems on a speaker-disjoint JSONL manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from koemo.asr import (  # noqa: E402
    AccuracyGateConfig,
    BenchmarkSplit,
    BenchmarkUtterance,
    NBestCandidate,
    SystemPrediction,
    aggregate_system_metrics,
    assert_manifest_integrity,
    evaluate_accuracy_gate,
    evaluate_system,
    paired_speaker_bootstrap,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
            records.append(value)
    return records


def _optional_string(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string or null")
    return value


def _required_string(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _parse_manifest(records: Iterable[dict[str, Any]]) -> tuple[BenchmarkUtterance, ...]:
    items: list[BenchmarkUtterance] = []
    for record in records:
        groups = record.get("groups", {})
        if not isinstance(groups, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in groups.items()
        ):
            raise TypeError("groups must be an object of string values")
        items.append(
            BenchmarkUtterance(
                utterance_id=_required_string(record, "utteranceId"),
                speaker_id=_required_string(record, "speakerId"),
                split=BenchmarkSplit(_required_string(record, "split")),
                reference_text=_required_string(record, "referenceText"),
                reference_reading=_optional_string(
                    record.get("referenceReading"),
                    field_name="referenceReading",
                ),
                target_reading=_optional_string(
                    record.get("targetReading"),
                    field_name="targetReading",
                ),
                observed_reading=_optional_string(
                    record.get("observedReading"),
                    field_name="observedReading",
                ),
                audio_sha256=_optional_string(
                    record.get("audioSha256"),
                    field_name="audioSha256",
                ),
                groups=groups,
            )
        )
    return tuple(items)


def _parse_prediction_candidate(record: dict[str, Any]) -> NBestCandidate:
    score = record.get("score")
    if score is not None and not isinstance(score, (int, float)):
        raise TypeError("candidate score must be numeric or null")
    text = record.get("text", "")
    if not isinstance(text, str):
        raise TypeError("candidate text must be a string")
    return NBestCandidate(
        candidate_id=_required_string(record, "candidateId"),
        text=text,
        reading=_optional_string(record.get("reading"), field_name="candidate reading"),
        score=None if score is None else float(score),
    )


def _parse_predictions(
    records: Iterable[dict[str, Any]],
    *,
    default_system_id: str,
) -> tuple[SystemPrediction, ...]:
    items: list[SystemPrediction] = []
    for record in records:
        candidate_records = record.get("candidates", [])
        if not isinstance(candidate_records, list) or not all(
            isinstance(value, dict) for value in candidate_records
        ):
            raise TypeError("candidates must be an array of objects")
        latency = record.get("latencyMs")
        if latency is not None and not isinstance(latency, (int, float)):
            raise TypeError("latencyMs must be numeric or null")
        text = record.get("text", "")
        if not isinstance(text, str):
            raise TypeError("prediction text must be a string")
        system_id = record.get("systemId", default_system_id)
        if not isinstance(system_id, str) or not system_id:
            raise ValueError("systemId must be a non-empty string")
        items.append(
            SystemPrediction(
                utterance_id=_required_string(record, "utteranceId"),
                system_id=system_id,
                text=text,
                reading=_optional_string(
                    record.get("reading"), field_name="prediction reading"
                ),
                status=_required_string(record, "status")
                if "status" in record
                else "accept",
                latency_ms=None if latency is None else float(latency),
                candidates=tuple(
                    _parse_prediction_candidate(value) for value in candidate_records
                ),
            )
        )
    return tuple(items)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate baseline and candidate Japanese ASR JSONL predictions with "
            "CER, mora metrics, oracle headroom, speaker bootstrap, and gates."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--split",
        default=BenchmarkSplit.TEST.value,
        choices=[value.value for value in BenchmarkSplit],
    )
    parser.add_argument("--baseline-id", default="baseline")
    parser.add_argument("--candidate-id", default="candidate")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--max-cer-regression", type=float, default=0.0)
    parser.add_argument("--max-mora-regression", type=float, default=0.0)
    parser.add_argument("--min-preservation-delta", type=float, default=0.0)
    parser.add_argument("--max-normalization-increase", type=float, default=0.0)
    parser.add_argument("--require-cer-ci", action="store_true")
    parser.add_argument("--require-mora-ci", action="store_true")
    parser.add_argument("--allow-speaker-overlap", action="store_true")
    parser.add_argument("--allow-audio-overlap", action="store_true")
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="exit with status 2 when an accuracy gate fails",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = _parse_manifest(_read_jsonl(args.manifest))
    assert_manifest_integrity(
        manifest,
        require_speaker_disjoint=not args.allow_speaker_overlap,
        require_audio_disjoint=not args.allow_audio_overlap,
    )

    split = BenchmarkSplit(args.split)
    selected_manifest = tuple(item for item in manifest if item.split is split)
    if not selected_manifest:
        raise ValueError(f"manifest contains no utterances in split {split.value!r}")

    baseline_predictions = _parse_predictions(
        _read_jsonl(args.baseline),
        default_system_id=args.baseline_id,
    )
    candidate_predictions = _parse_predictions(
        _read_jsonl(args.candidate),
        default_system_id=args.candidate_id,
    )
    selected_ids = {item.utterance_id for item in selected_manifest}
    baseline_predictions = tuple(
        item for item in baseline_predictions if item.utterance_id in selected_ids
    )
    candidate_predictions = tuple(
        item for item in candidate_predictions if item.utterance_id in selected_ids
    )

    baseline_evaluations = evaluate_system(selected_manifest, baseline_predictions)
    candidate_evaluations = evaluate_system(selected_manifest, candidate_predictions)
    baseline_metrics = aggregate_system_metrics(baseline_evaluations)
    candidate_metrics = aggregate_system_metrics(candidate_evaluations)

    cer_bootstrap = paired_speaker_bootstrap(
        baseline_evaluations,
        candidate_evaluations,
        metric="cer",
        samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
        seed=args.bootstrap_seed,
    )
    mora_bootstrap = None
    if (
        baseline_metrics.mora_evaluated == len(selected_manifest)
        and candidate_metrics.mora_evaluated == len(selected_manifest)
    ):
        mora_bootstrap = paired_speaker_bootstrap(
            baseline_evaluations,
            candidate_evaluations,
            metric="mora_error_rate",
            samples=args.bootstrap_samples,
            confidence_level=args.confidence_level,
            seed=args.bootstrap_seed,
        )

    gate = evaluate_accuracy_gate(
        baseline_metrics,
        candidate_metrics,
        config=AccuracyGateConfig(
            max_cer_regression=args.max_cer_regression,
            max_mora_regression=args.max_mora_regression,
            min_learner_preservation_delta=args.min_preservation_delta,
            max_normalized_to_target_increase=args.max_normalization_increase,
            require_cer_confidence_improvement=args.require_cer_ci,
            require_mora_confidence_improvement=args.require_mora_ci,
        ),
        cer_bootstrap=cer_bootstrap,
        mora_bootstrap=mora_bootstrap,
    )
    report = {
        "schemaVersion": "koemo-mora-asr-benchmark-v1",
        "split": split.value,
        "utteranceCount": len(selected_manifest),
        "speakerCount": len({item.speaker_id for item in selected_manifest}),
        "baseline": baseline_metrics.to_dict(),
        "candidate": candidate_metrics.to_dict(),
        "comparisons": {
            "cer": cer_bootstrap.to_dict(),
            "moraErrorRate": (
                None if mora_bootstrap is None else mora_bootstrap.to_dict()
            ),
        },
        "gate": gate.to_dict(),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 2 if args.fail_on_regression and not gate.passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
