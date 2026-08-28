from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import os
from pathlib import Path


def pseudonymize(client_id: str, salt: bytes) -> str:
    return "spk_" + hmac.new(salt, client_id.encode("utf-8"), hashlib.sha256).hexdigest()[:24]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a local Common Voice manifest without copying audio or exporting client_id."
    )
    parser.add_argument("tsv", type=Path)
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-id", default="common-voice-26-ja")
    parser.add_argument("--salt-env", default="MORAWEAVE_SPEAKER_HMAC_SALT")
    args = parser.parse_args()

    salt_text = os.environ.get(args.salt_env)
    if not salt_text or len(salt_text.encode("utf-8")) < 16:
        raise SystemExit(f"{args.salt_env} must contain at least 16 UTF-8 bytes")
    salt = salt_text.encode("utf-8")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    input_hash = hashlib.sha256()
    rows_written = 0
    speakers: set[str] = set()
    with args.tsv.open("rb") as raw:
        for chunk in iter(lambda: raw.read(1024 * 1024), b""):
            input_hash.update(chunk)

    with args.tsv.open(encoding="utf-8", newline="") as source, args.output.open("w", encoding="utf-8") as target:
        reader = csv.DictReader(source, delimiter="\t")
        required = {"path", "sentence", "client_id"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise SystemExit(f"TSV must contain: {', '.join(sorted(required))}")
        for row in reader:
            sentence = (row.get("sentence") or "").strip()
            relative = (row.get("path") or "").strip()
            client_id = (row.get("client_id") or "").strip()
            if not sentence or not relative or not client_id:
                continue
            audio_path = (args.clips_dir / relative).resolve()
            speaker = pseudonymize(client_id, salt)
            speakers.add(speaker)
            payload = {
                "assetId": args.asset_id,
                "audioPath": str(audio_path),
                "transcript": sentence,
                "speakerPseudonym": speaker,
                "speakerIdExported": False,
                "sourceClientIdPresent": False,
            }
            target.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            rows_written += 1

    report = {
        "assetId": args.asset_id,
        "sourceTsvSha256": input_hash.hexdigest(),
        "rowsWritten": rows_written,
        "speakerPseudonyms": len(speakers),
        "audioCopied": False,
        "rawSpeakerIdsExported": False,
    }
    report_path = args.output.with_suffix(args.output.suffix + ".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
