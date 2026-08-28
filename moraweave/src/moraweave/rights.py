from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Permission = Literal["allow", "deny", "review"]


@dataclass(frozen=True, slots=True)
class RightsRecord:
    asset_id: str
    source_name: str
    source_url: str
    license_name: str
    license_url: str
    train: Permission
    derive_features: Permission
    redistribute_raw: Permission
    export_speaker_id: Permission
    attribution: str
    reviewed_at: str
    notes: str = ""

    @classmethod
    def from_dict(cls, row: dict[str, object]) -> "RightsRecord":
        return cls(
            asset_id=str(row["assetId"]),
            source_name=str(row["sourceName"]),
            source_url=str(row["sourceUrl"]),
            license_name=str(row["licenseName"]),
            license_url=str(row["licenseUrl"]),
            train=str(row["train"]),  # type: ignore[arg-type]
            derive_features=str(row["deriveFeatures"]),  # type: ignore[arg-type]
            redistribute_raw=str(row["redistributeRaw"]),  # type: ignore[arg-type]
            export_speaker_id=str(row["exportSpeakerId"]),  # type: ignore[arg-type]
            attribution=str(row["attribution"]),
            reviewed_at=str(row["reviewedAt"]),
            notes=str(row.get("notes", "")),
        )


class RightsRegistry:
    def __init__(self, records: list[RightsRecord]) -> None:
        self._records = {record.asset_id: record for record in records}
        if len(self._records) != len(records):
            raise ValueError("rights asset IDs must be unique")

    @classmethod
    def load(cls, path: str | Path) -> "RightsRegistry":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("assets"), list):
            raise ValueError("rights registry must contain an assets array")
        return cls([RightsRecord.from_dict(row) for row in payload["assets"]])

    def require(self, asset_id: str, operation: Literal["train", "derive_features", "redistribute_raw", "export_speaker_id"]) -> RightsRecord:
        try:
            record = self._records[asset_id]
        except KeyError as exc:
            raise PermissionError(f"asset is absent from rights registry: {asset_id}") from exc
        permission = getattr(record, operation)
        if permission != "allow":
            raise PermissionError(f"{operation} is {permission} for {asset_id}")
        return record

    def export_attributions(self) -> list[str]:
        return sorted({record.attribution for record in self._records.values() if record.attribution})


def pseudonymize_speaker(speaker_id: str, secret_salt: bytes) -> str:
    if len(secret_salt) < 16:
        raise ValueError("speaker HMAC salt must contain at least 16 bytes")
    digest = hmac.new(secret_salt, speaker_id.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"spk_{digest[:24]}"


def safe_manifest_row(
    *,
    asset_id: str,
    audio_path: str,
    transcript: str,
    speaker_id: str | None,
    secret_salt: bytes,
    registry: RightsRegistry,
) -> dict[str, object]:
    registry.require(asset_id, "train")
    pseudonym = None
    if speaker_id:
        pseudonym = pseudonymize_speaker(speaker_id, secret_salt)
    return {
        "assetId": asset_id,
        "audioPath": audio_path,
        "transcript": transcript,
        "speakerPseudonym": pseudonym,
        "speakerIdExported": False,
    }
