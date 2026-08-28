#!/usr/bin/env python3
"""Japanese kana normalization, mora segmentation, and character-span merging."""

from __future__ import annotations

import argparse
import json
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Iterable

SMALL_KANA = frozenset("ァィゥェォャュョヮヵヶ")
PUNCTUATION = frozenset(" \t\r\n、。,.!?！？・「」『』（）()［］[]【】…‥:：;；")


@dataclass(slots=True)
class MoraUnit:
    index: int
    surface: str
    kana: str
    type: str
    start_ms: float | None = None
    end_ms: float | None = None
    confidence: float | None = None
    source: str = "text"


def to_katakana(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "")
    output: list[str] = []
    for char in normalized:
        code = ord(char)
        output.append(chr(code + 0x60) if 0x3041 <= code <= 0x3096 else char)
    return "".join(output)


def classify_mora(kana: str) -> str:
    if kana == "ン":
        return "moraic-nasal"
    if kana == "ッ":
        return "geminate"
    if kana == "ー":
        return "long-vowel"
    return "regular"


def _is_katakana(char: str) -> bool:
    return len(char) == 1 and 0x30A0 <= ord(char) <= 0x30FF


def split_mora(value: str, *, include_unknown: bool = False) -> list[MoraUnit]:
    units: list[MoraUnit] = []
    for char in to_katakana(value):
        if char in PUNCTUATION:
            continue
        if char in SMALL_KANA and units and units[-1].type == "regular":
            units[-1].surface += char
            units[-1].kana += char
            continue
        if not _is_katakana(char) and not include_unknown:
            continue
        units.append(
            MoraUnit(
                index=len(units),
                surface=char,
                kana=char,
                type=classify_mora(char),
            )
        )
    return units


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _confidence(value: Any) -> float | None:
    number = _number(value)
    return None if number is None else max(0.0, min(1.0, number))


def merge_character_alignment(character_units: Iterable[dict[str, Any]]) -> list[MoraUnit]:
    mora_units: list[MoraUnit] = []

    for item in character_units:
        raw = item.get("char", item.get("surface", item.get("text", "")))
        for char in to_katakana(str(raw)):
            if char in PUNCTUATION or not _is_katakana(char):
                continue

            start_ms = _number(item.get("startMs", item.get("start_ms")))
            end_ms = _number(item.get("endMs", item.get("end_ms")))
            confidence = _confidence(item.get("confidence"))

            if char in SMALL_KANA and mora_units and mora_units[-1].type == "regular":
                previous = mora_units[-1]
                previous.surface += char
                previous.kana += char
                if previous.start_ms is None:
                    previous.start_ms = start_ms
                if end_ms is not None:
                    previous.end_ms = end_ms
                if confidence is not None:
                    previous.confidence = (
                        confidence
                        if previous.confidence is None
                        else min(previous.confidence, confidence)
                    )
                continue

            mora_units.append(
                MoraUnit(
                    index=len(mora_units),
                    surface=char,
                    kana=char,
                    type=classify_mora(char),
                    start_ms=start_ms,
                    end_ms=end_ms,
                    confidence=confidence,
                    source="char-merge",
                )
            )

    return mora_units


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?")
    parser.add_argument("--alignment-json")
    args = parser.parse_args()

    if args.alignment_json:
        with open(args.alignment_json, encoding="utf-8") as handle:
            units = merge_character_alignment(json.load(handle))
    elif args.text is not None:
        units = split_mora(args.text)
    else:
        parser.error("text or --alignment-json is required")

    print(json.dumps([asdict(unit) for unit in units], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
