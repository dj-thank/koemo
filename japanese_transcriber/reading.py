from __future__ import annotations

import re

_KANA_ONLY = re.compile(r"^[\u3040-\u30ffー\s、。！？!?・]+$")


def _mora_split(text: str) -> tuple[str | None, list[str]]:
    try:
        from scripts.japanese_mora import split_mora
    except ImportError:
        return None, []
    units = split_mora(text)
    return text, [unit.kana for unit in units]


def reading_and_mora(text: str, *, use_pyopenjtalk: bool = False) -> tuple[str | None, list[str]]:
    value = str(text or "").strip()
    if not value:
        return None, []
    if _KANA_ONLY.fullmatch(value):
        return _mora_split(value)
    if not use_pyopenjtalk:
        return None, []
    try:
        import pyopenjtalk
    except ImportError:
        return None, []
    try:
        reading = pyopenjtalk.g2p(value, kana=True)
    except Exception:
        return None, []
    if not reading:
        return None, []
    _, mora = _mora_split(reading)
    return reading, mora
