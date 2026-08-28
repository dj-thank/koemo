from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

PERMISSIONS = {"allow", "deny", "review"}
REQUIRED = {
    "assetId", "sourceName", "sourceUrl", "licenseName", "licenseUrl",
    "train", "deriveFeatures", "redistributeRaw", "exportSpeakerId",
    "attribution", "reviewedAt"
}


def validate(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    assets = payload.get("assets") if isinstance(payload, dict) else None
    if not isinstance(assets, list):
        return ["registry must contain an assets array"]
    ids: set[str] = set()
    for index, row in enumerate(assets):
        prefix = f"assets[{index}]"
        if not isinstance(row, dict):
            errors.append(f"{prefix} must be an object")
            continue
        missing = REQUIRED - row.keys()
        if missing:
            errors.append(f"{prefix} missing: {', '.join(sorted(missing))}")
        asset_id = str(row.get("assetId", ""))
        if not asset_id:
            errors.append(f"{prefix}.assetId is empty")
        elif asset_id in ids:
            errors.append(f"duplicate assetId: {asset_id}")
        ids.add(asset_id)
        for field in ("train", "deriveFeatures", "redistributeRaw", "exportSpeakerId"):
            if row.get(field) not in PERMISSIONS:
                errors.append(f"{prefix}.{field} must be allow, deny, or review")
        for field in ("sourceUrl", "licenseUrl"):
            parsed = urlparse(str(row.get(field, "")))
            if parsed.scheme != "https" or not parsed.netloc:
                errors.append(f"{prefix}.{field} must be an https URL")
        if row.get("redistributeRaw") == "allow" and not row.get("attribution"):
            errors.append(f"{prefix} permits redistribution without attribution metadata")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("registry", type=Path)
    args = parser.parse_args()
    errors = validate(args.registry)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("rights registry: valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
