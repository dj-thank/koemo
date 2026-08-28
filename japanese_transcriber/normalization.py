from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

_JAPANESE_OR_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff]", re.UNICODE)
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([、。！？!?：:；;）\]】」』])")
_SPACE_AFTER_OPEN = re.compile(r"([（\[【「『])\s+")
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def _is_ascii_word_char(char: str) -> bool:
    return bool(char) and char.isascii() and (char.isalnum() or char in "_+-/#@.%")


def _needs_boundary_space(left: str, right: str) -> bool:
    if not left or not right:
        return False
    return _is_ascii_word_char(left[-1]) and _is_ascii_word_char(right[0])


def join_fragments(fragments: Iterable[str]) -> str:
    """Join Whisper segments without inserting spaces into Japanese prose."""

    output = ""
    for raw in fragments:
        fragment = str(raw or "").strip()
        if not fragment:
            continue
        if output and _needs_boundary_space(output, fragment):
            output += " "
        output += fragment
    return output


def deterministic_normalize(text: str) -> str:
    """Create a readable derivative while leaving observed text untouched."""

    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = _MULTI_SPACE.sub(" ", value)
    value = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", value)
    value = _SPACE_AFTER_OPEN.sub(r"\1", value)
    if contains_japanese(value):
        value = value.replace("!", "！").replace("?", "？")
    value = re.sub(r"([、。！？])\1{2,}", r"\1\1", value)
    value = _MULTI_NEWLINE.sub("\n\n", value)
    return value.strip()


def contains_japanese(text: str) -> bool:
    return bool(_JAPANESE_OR_CJK.search(str(text or "")))
