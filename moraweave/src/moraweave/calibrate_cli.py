from __future__ import annotations

import argparse
import json
from pathlib import Path

from .calibration import (
    CalibrationProfile,
    brier_score,
    expected_calibration_error,
    fit_temperature,
    negative_log_likelihood,
    risk_coverage_curve,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moraweave-calibrate",
        description="Fit scalar confidence temperature on a held-out labelled JSONL set.",
    )
    parser.add_argument("input", help="JSONL rows with confidence and correct fields")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--name", default="observed-posterior")
    parser.add_argument("--bins", type=int, default=15)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    probabilities: list[float] = []
    labels: list[int] = []
    for line_number, line in enumerate(Path(args.input).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        try:
            probability = float(row["confidence"])
            correct = int(bool(row["correct"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid calibration row {line_number}") from exc
        if not 0 <= probability <= 1:
            raise ValueError(f"confidence outside [0, 1] on line {line_number}")
        probabilities.append(probability)
        labels.append(correct)
    if len(probabilities) < 10:
        raise ValueError("at least ten held-out examples are required")

    temperature = fit_temperature(probabilities, labels)
    profile = CalibrationProfile(
        name=args.name,
        temperature=temperature,
        input_kind="probability",
    )
    calibrated = [profile.transform(value) for value in probabilities]
    assert all(value is not None for value in calibrated)
    calibrated_values = [float(value) for value in calibrated if value is not None]
    payload = {
        "schemaVersion": "1.0.0",
        "profile": {
            "name": profile.name,
            "center": profile.center,
            "scale": profile.scale,
            "temperature": profile.temperature,
            "direction": profile.direction,
            "inputKind": profile.input_kind,
            "version": profile.version,
            "digest": profile.digest,
        },
        "sampleCount": len(labels),
        "before": {
            "ece": expected_calibration_error(probabilities, labels, bins=args.bins),
            "brier": brier_score(probabilities, labels),
            "nll": negative_log_likelihood(probabilities, labels),
        },
        "after": {
            "ece": expected_calibration_error(calibrated_values, labels, bins=args.bins),
            "brier": brier_score(calibrated_values, labels),
            "nll": negative_log_likelihood(calibrated_values, labels),
        },
        "riskCoverage": [
            {"coverage": coverage, "risk": risk}
            for coverage, risk in risk_coverage_curve(calibrated_values, labels)
        ],
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(target), "temperature": temperature}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
