"""Koemo用途の日本語認識補正。

公開データや手元ログから生成したJSONを `~/.koemo/native_corrections.json`
に置くと、内蔵辞書に重ねて使う。Windows native live と Whisper final の
両方で、Koemo固有語や短文テストの頻出誤変換だけを補正する。
"""
import json
import re
from functools import lru_cache
from pathlib import Path

from .config import CONFIG_DIR


BUILTIN_FILE = Path(__file__).resolve().parent / "data" / "native_corrections.json"
USER_FILE = CONFIG_DIR / "native_corrections.json"


@lru_cache(maxsize=1)
def load_native_corrections():
    merged = {"regex": [], "drop_if_alone": [], "grammar_phrases": []}
    for path in (BUILTIN_FILE, USER_FILE):
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        merged["regex"].extend(data.get("regex") or [])
        merged["drop_if_alone"].extend(data.get("drop_if_alone") or [])
        merged["grammar_phrases"].extend(data.get("grammar_phrases") or [])
    return merged


def grammar_phrases(limit=100):
    """Windows Speech へ渡す短い語彙ヒント。過剰な一般文は渡さない。"""
    seen = set()
    out = []
    for raw in load_native_corrections().get("grammar_phrases", []):
        phrase = str(raw).strip()
        if not phrase or phrase in seen:
            continue
        if len(phrase) < 3 or len(phrase) > 28:
            continue
        seen.add(phrase)
        out.append(phrase)
        if len(out) >= limit:
            break
    return out


def normalize_native_text(text):
    """Koemo用途で頻出する日本語ASRの誤変換を補正する。"""
    text = (text or "").strip()
    if not text:
        return ""
    data = load_native_corrections()
    for item in data.get("regex", []):
        if not isinstance(item, list) or len(item) != 2:
            continue
        pattern, repl = item
        try:
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        except re.error:
            continue
    return text.strip()


normalize_transcript_text = normalize_native_text


def filter_native_texts(texts):
    """単独のノイズ断片を落とす。文中の助詞までは落とさない。"""
    data = load_native_corrections()
    drop = set(str(x) for x in data.get("drop_if_alone", []))
    out = []
    for text in texts:
        cleaned = normalize_native_text(text)
        if not cleaned:
            continue
        if cleaned in drop:
            continue
        out.append(cleaned)
    return out
