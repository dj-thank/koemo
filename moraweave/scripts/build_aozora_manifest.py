from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a per-work Aozora manifest; no corpus-wide permission is inferred."
    )
    parser.add_argument("input", type=Path, help="JSONL: workId,title,path,rightsStatus,sourceUrl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, str]] = []
    for line_number, line in enumerate(args.input.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        status = str(row.get("rightsStatus", "review"))
        if status != "allow-feature-derivation":
            rejected.append({"workId": str(row.get("workId", "")), "reason": status})
            continue
        path = Path(str(row["path"]))
        if not path.is_file():
            rejected.append({"workId": str(row.get("workId", "")), "reason": "missing-file"})
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        accepted.append(
            {
                "assetId": "aozora-work-manifest",
                "workId": str(row["workId"]),
                "title": str(row.get("title", "")),
                "localPath": str(path.resolve()),
                "sourceUrl": str(row.get("sourceUrl", "")),
                "inputSha256": digest,
                "deriveFeatures": True,
                "redistributeRaw": False,
            }
        )

    payload = {
        "schemaVersion": "1.0.0",
        "acceptedWorks": accepted,
        "rejectedWorks": rejected,
        "corpusWidePermissionAssumed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"accepted": len(accepted), "rejected": len(rejected)}, ensure_ascii=False))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
