"""Japanese mora normalization and character-CTC aggregation.

The module intentionally works from a *reading*.  Kanji-to-reading conversion is
an upstream linguistic task and must not be guessed here.  Long vowels (ー), the
moraic nasal (ン), and the sokuon (ッ) each remain one mora.  Small kana such as
ャ/ュ/ョ are attached to the preceding kana.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
import unicodedata
from typing import Iterable, Sequence

from .schema import CTCCharUnit, MoraUnit, TextSpan, TimeSpan, UnitKind, UnitSource


# Small kana that modify the preceding kana.  Small tsu is deliberately absent:
# it is an independent mora.
_SMALL_ATTACH = frozenset(
    "ァィゥェォャュョヮヵヶ"
    "ㇰㇱㇲㇳㇴㇵㇶㇷㇸㇹㇺㇻㇼㇽㇾㇿ"
)
_COMBINING_VOICING = frozenset(("\u3099", "\u309a"))
_MORA_SPECIALS = frozenset(("ッ", "ン", "ー"))

_BOUNDARIES = frozenset(
    " \t\r\n、。,.，．!！?？:：;；・/／|｜()（）[]［］{}｛｝「」『』【】〈〉《》“”\"'…‥—―-"
)
_NOISE_TOKENS = frozenset(
    {
        "<NOISE>",
        "[NOISE]",
        "<SIL>",
        "[SIL]",
        "<UNK>",
        "[UNK]",
        "<LAUGH>",
        "[LAUGH]",
    }
)


def normalize_kana(text: str) -> str:
    """NFKC-normalize text and convert hiragana to katakana.

    Katakana is the canonical reading script used by ``moraUnits``.  Non-kana
    characters are preserved so that code-switching and unknown tokens remain
    observable rather than disappearing.
    """

    normalized = unicodedata.normalize("NFKC", text)
    chars: list[str] = []
    for char in normalized:
        code = ord(char)
        if 0x3041 <= code <= 0x3096:
            chars.append(chr(code + 0x60))
        else:
            chars.append(char)
    return "".join(chars)


def _is_katakana(char: str) -> bool:
    if not char:
        return False
    code = ord(char)
    return 0x30A1 <= code <= 0x30FA or char in _MORA_SPECIALS or char in _SMALL_ATTACH


def _geometric_mean(values: Sequence[float]) -> float:
    if not values:
        return 1.0
    # A true zero posterior should keep the merged unit at zero.
    if any(value <= 0.0 for value in values):
        return 0.0
    return exp(sum(log(value) for value in values) / len(values))


def _merged_time(units: Sequence[CTCCharUnit]) -> TimeSpan | None:
    if not units or any(unit.start is None or unit.end is None for unit in units):
        return None
    starts = [unit.start for unit in units if unit.start is not None]
    ends = [unit.end for unit in units if unit.end is not None]
    return TimeSpan(min(starts), max(ends))


def _merged_frame_span(units: Sequence[CTCCharUnit]) -> tuple[int | None, int | None]:
    if not units or any(unit.frame_start is None or unit.frame_end is None for unit in units):
        return None, None
    starts = [unit.frame_start for unit in units if unit.frame_start is not None]
    ends = [unit.frame_end for unit in units if unit.frame_end is not None]
    return min(starts), max(ends)


def collapse_ctc_path(path: Iterable[CTCCharUnit]) -> tuple[CTCCharUnit, ...]:
    """Apply standard CTC collapse to a frame/token path.

    Consecutive duplicate non-blank symbols are merged.  A blank flushes the
    active run, so the same symbol after a blank remains a distinct output unit.
    Posterior is aggregated with a geometric mean and the full time/frame span is
    retained.
    """

    output: list[CTCCharUnit] = []
    run: list[CTCCharUnit] = []
    run_symbol: str | None = None

    def flush() -> None:
        nonlocal run, run_symbol
        if not run:
            return
        time_span = _merged_time(run)
        frame_start, frame_end = _merged_frame_span(run)
        token_ids = {unit.token_id for unit in run}
        output.append(
            CTCCharUnit(
                symbol=run_symbol or run[0].symbol,
                posterior=_geometric_mean([unit.posterior for unit in run]),
                start=None if time_span is None else time_span.start,
                end=None if time_span is None else time_span.end,
                token_id=next(iter(token_ids)) if len(token_ids) == 1 else None,
                frame_start=frame_start,
                frame_end=frame_end,
                is_blank=False,
            )
        )
        run = []
        run_symbol = None

    for unit in path:
        if unit.is_blank:
            flush()
            continue
        normalized_symbol = normalize_kana(unit.symbol)
        if run and normalized_symbol == run_symbol:
            run.append(unit)
            continue
        flush()
        run = [unit]
        run_symbol = normalized_symbol
    flush()
    return tuple(output)


@dataclass(slots=True)
class _MutableUnit:
    surface: str
    reading: str
    kind: UnitKind
    text_start: int
    text_end: int
    constituents: list[CTCCharUnit]
    source_indices: list[int]

    def append(
        self,
        *,
        surface: str,
        reading: str,
        text_end: int,
        constituent: CTCCharUnit,
        source_index: int,
    ) -> None:
        self.surface += surface
        self.reading += reading
        self.text_end = text_end
        self.constituents.append(constituent)
        self.source_indices.append(source_index)


def merge_char_ctc_to_mora(
    units: Iterable[CTCCharUnit],
    *,
    keep_boundaries: bool = False,
    source: UnitSource = UnitSource.CHAR_CTC,
) -> tuple[MoraUnit, ...]:
    """Merge collapsed character CTC units into canonical ``MoraUnit`` objects.

    The input may contain a multi-character symbol; it is expanded while
    retaining the token's timing/posterior.  The function does not run a
    morphological analyzer and therefore treats non-kana symbols as ``OTHER``.
    """

    mutable: list[_MutableUnit] = []
    text_position = 0

    for source_index, token in enumerate(units):
        if token.is_blank:
            continue

        normalized_token = normalize_kana(token.symbol)
        if normalized_token.upper() in _NOISE_TOKENS:
            token_length = max(1, len(normalized_token))
            mutable.append(
                _MutableUnit(
                    surface=token.symbol,
                    reading=normalized_token,
                    kind=UnitKind.NOISE,
                    text_start=text_position,
                    text_end=text_position + token_length,
                    constituents=[token],
                    source_indices=[source_index],
                )
            )
            text_position += token_length
            continue

        # NFKC can change length.  Spans intentionally refer to the canonical
        # reading stream, which is the common representation used downstream.
        raw_chars = list(unicodedata.normalize("NFKC", token.symbol))
        canonical_chars = list(normalized_token)
        if len(raw_chars) != len(canonical_chars):
            raw_chars = canonical_chars

        for char_index, canonical_char in enumerate(canonical_chars):
            surface_char = raw_chars[char_index]
            start = text_position
            text_position += 1

            if canonical_char in _BOUNDARIES or canonical_char.isspace():
                if keep_boundaries:
                    mutable.append(
                        _MutableUnit(
                            surface=surface_char,
                            reading=canonical_char,
                            kind=UnitKind.BOUNDARY,
                            text_start=start,
                            text_end=text_position,
                            constituents=[token],
                            source_indices=[source_index],
                        )
                    )
                continue

            if canonical_char in _COMBINING_VOICING:
                if mutable and mutable[-1].kind is UnitKind.MORA:
                    mutable[-1].append(
                        surface=surface_char,
                        reading=canonical_char,
                        text_end=text_position,
                        constituent=token,
                        source_index=source_index,
                    )
                else:
                    mutable.append(
                        _MutableUnit(
                            surface=surface_char,
                            reading=canonical_char,
                            kind=UnitKind.OTHER,
                            text_start=start,
                            text_end=text_position,
                            constituents=[token],
                            source_indices=[source_index],
                        )
                    )
                continue

            if canonical_char in _SMALL_ATTACH:
                if mutable and mutable[-1].kind is UnitKind.MORA:
                    mutable[-1].append(
                        surface=surface_char,
                        reading=canonical_char,
                        text_end=text_position,
                        constituent=token,
                        source_index=source_index,
                    )
                else:
                    # Preserve malformed/isolated small kana rather than
                    # dropping evidence.  It is still counted as a mora unit.
                    mutable.append(
                        _MutableUnit(
                            surface=surface_char,
                            reading=canonical_char,
                            kind=UnitKind.MORA,
                            text_start=start,
                            text_end=text_position,
                            constituents=[token],
                            source_indices=[source_index],
                        )
                    )
                continue

            kind = UnitKind.MORA if _is_katakana(canonical_char) else UnitKind.OTHER
            mutable.append(
                _MutableUnit(
                    surface=surface_char,
                    reading=canonical_char,
                    kind=kind,
                    text_start=start,
                    text_end=text_position,
                    constituents=[token],
                    source_indices=[source_index],
                )
            )

    result: list[MoraUnit] = []
    counters = {
        UnitKind.MORA: 0,
        UnitKind.BOUNDARY: 0,
        UnitKind.NOISE: 0,
        UnitKind.OTHER: 0,
    }
    prefixes = {
        UnitKind.MORA: "m",
        UnitKind.BOUNDARY: "b",
        UnitKind.NOISE: "n",
        UnitKind.OTHER: "o",
    }

    for item in mutable:
        time_span = _merged_time(item.constituents)
        posterior = _geometric_mean([unit.posterior for unit in item.constituents])
        ordinal = counters[item.kind]
        counters[item.kind] += 1
        result.append(
            MoraUnit(
                unit_id=f"{prefixes[item.kind]}{ordinal:04d}",
                surface=item.surface,
                reading=item.reading,
                mora=item.reading if item.kind is UnitKind.MORA else "",
                kind=item.kind,
                source=source,
                text_span=TextSpan(item.text_start, item.text_end),
                time_span=time_span,
                posterior=posterior,
                source_indices=tuple(dict.fromkeys(item.source_indices)),
                metadata={"constituentCount": len(item.constituents)},
            )
        )
    return tuple(result)


def mora_units_from_reading(
    reading: str,
    *,
    keep_boundaries: bool = False,
    source: UnitSource = UnitSource.TEXT_READING,
) -> tuple[MoraUnit, ...]:
    """Create canonical units from a known reading string."""

    original = unicodedata.normalize("NFKC", reading)
    char_units = [
        CTCCharUnit(symbol=char, posterior=1.0)
        for char in original
    ]
    return merge_char_ctc_to_mora(
        char_units,
        keep_boundaries=keep_boundaries,
        source=source,
    )


def split_mora(reading: str) -> tuple[str, ...]:
    """Return mora labels only, excluding boundaries and unknown symbols."""

    return tuple(
        unit.mora
        for unit in mora_units_from_reading(reading)
        if unit.kind is UnitKind.MORA
    )


def mora_count(reading: str) -> int:
    return len(split_mora(reading))
