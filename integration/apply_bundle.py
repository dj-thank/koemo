#!/usr/bin/env python3
"""Install the additive mora-aware ASR modules into an existing PoC repository.

This installer deliberately does not rewrite the PoC's existing orchestration,
scoring, fluency, or GOP files without seeing their exact syntax. It copies only
new, independently tested files, verifies the copied bytes, and reports the
integration points that still need to be wired.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
COPY_PATHS = (
    # Node runtime.
    "src/mora.mjs",
    "src/transcript-contract.mjs",
    "src/asr-fusion.mjs",
    "src/mora-ctc-fusion.mjs",
    "src/local-lm-reranker.mjs",
    "src/ollama-candidate-reranker.mjs",
    "src/transcript-pipeline.mjs",
    "src/fluency-from-mora.mjs",
    # Python runtime and N-best adapter.
    "scripts/__init__.py",
    "scripts/japanese_mora.py",
    "scripts/merge_mora_alignment.py",
    "scripts/whisper_nbest.py",
    "scripts/ollama_rerank.mjs",
    # Whisper architecture/training layer.
    "training/__init__.py",
    "training/mora_vocab.py",
    "training/mora_ctc_runtime.py",
    "training/whisper_mora_multitask.py",
    # Contracts and documentation.
    "schemas/mora-unit.schema.json",
    "schemas/transcript-record.schema.json",
    "schemas/asr-nbest.schema.json",
    "schemas/mora-multitask-config.schema.json",
    "SSOT_MORA.md",
    "docs/UPSTREAM_COMPATIBILITY.md",
    "integration/INTEGRATION.md",
)
EXPECTED_INTEGRATION_FILES = (
    "server.mjs",
    "src/fluency.mjs",
    "src/scoring.mjs",
    "scripts/gop_align.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, help="existing japanese-speaking-assessment-poc root")
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite only the additive mora-core paths listed by this installer",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show the copy/skip plan without changing the target repository",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    target = args.target.expanduser().resolve()
    if not target.is_dir():
        sys.stderr.write(f"error: target repository does not exist: {target}\n")
        return 2

    copied = 0
    identical = 0
    skipped = 0
    planned = 0

    for relative in COPY_PATHS:
        source = BUNDLE_ROOT / relative
        if not source.is_file():
            sys.stderr.write(f"error: bundle is incomplete; missing source: {source}\n")
            return 3

        destination = target / relative
        source_hash = sha256(source)
        if destination.is_file() and sha256(destination) == source_hash:
            print(f"SAME: {relative}")
            identical += 1
            continue
        if destination.exists() and not args.force:
            print(f"SKIP different existing file: {relative}")
            skipped += 1
            continue
        if args.dry_run:
            action = "REPLACE" if destination.exists() else "COPY"
            print(f"PLAN {action}: {relative}")
            planned += 1
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha256(destination) != source_hash:
            sys.stderr.write(f"error: post-copy SHA-256 mismatch: {relative}\n")
            return 4
        print(f"COPY: {relative}")
        copied += 1

    print("\nExisting PoC integration points:")
    missing = 0
    for relative in EXPECTED_INTEGRATION_FILES:
        exists = (target / relative).is_file()
        print(f"  {'FOUND' if exists else 'MISSING'} {relative}")
        missing += int(not exists)

    print(
        "\nDone: "
        f"copied={copied}, identical={identical}, skipped={skipped}, "
        f"planned={planned}, missingIntegrationFiles={missing}"
    )
    print(
        "Next: read SSOT_MORA.md and integration/INTEGRATION.md, then wire the "
        "reported PoC files using their real function names."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
