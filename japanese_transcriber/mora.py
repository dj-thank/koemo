from __future__ import annotations

import unicodedata
from dataclasses import dataclass

SMALL_KANA = frozenset("ァィゥェォャュョヮヵヶ")
PUNCTUATION = frozenset(" \t\r\n、。,.!?！？・「」『』（）()［］[]【】…‥:：;；")


@dataclass(slots=True)
class Mora:
    index: int
    surface: str
    kana: str
    type: str


def to_katakana(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    output: list[str] = []
    for char in normalized:
        codepoint = ord(char)
        output.append(chr(codepoint + 0x60) if 0x3041 <= codepoint <= 0x3096 else char)
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


def split_mora(value: str, *, include_unknown: bool = False) -> list[Mora]:
    units: list[Mora] = []
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
            Mora(
                index=len(units),
                surface=char,
                kana=char,
                type=classify_mora(char),
            )
        )
    return units


def count_mora(value: str) -> int:
    return len(split_mora(value))
