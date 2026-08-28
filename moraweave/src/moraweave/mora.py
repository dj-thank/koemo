from __future__ import annotations

import unicodedata
from dataclasses import dataclass

SMALL_KANA = frozenset("ァィゥェォャュョヮヵヶ")
PUNCTUATION = frozenset(" \t\r\n、。,.!?！？・「」『』（）()［］[]【】…‥:：;；")


@dataclass(frozen=True, slots=True)
class TextMora:
    kana: str
    start: int
    end: int
    kind: str


def to_katakana(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    output: list[str] = []
    for char in value:
        code = ord(char)
        output.append(chr(code + 0x60) if 0x3041 <= code <= 0x3096 else char)
    return "".join(output)


def mora_kind(kana: str) -> str:
    if kana == "ン":
        return "moraic-nasal"
    if kana == "ッ":
        return "geminate"
    if kana == "ー":
        return "long-vowel"
    return "regular"


def segment_mora(text: str, *, keep_unknown: bool = False) -> list[TextMora]:
    normalized = to_katakana(text)
    result: list[TextMora] = []
    for index, char in enumerate(normalized):
        if char in PUNCTUATION:
            continue
        is_katakana = 0x30A0 <= ord(char) <= 0x30FF
        if char in SMALL_KANA and result and result[-1].kind == "regular":
            previous = result[-1]
            result[-1] = TextMora(
                kana=previous.kana + char,
                start=previous.start,
                end=index + 1,
                kind=previous.kind,
            )
            continue
        if not is_katakana and not keep_unknown:
            continue
        result.append(TextMora(kana=char, start=index, end=index + 1, kind=mora_kind(char)))
    return result


def mora_sequence(text: str) -> list[str]:
    return [item.kana for item in segment_mora(text)]
