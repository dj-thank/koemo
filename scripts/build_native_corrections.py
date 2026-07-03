r"""公開データ/手元ログから Windows純正ASR用の補正JSONを作る。

使い方:
  python scripts\build_native_corrections.py --from-koemo-outputs --output %USERPROFILE%\.koemo\native_corrections.json
  python scripts\build_native_corrections.py --hf-dataset mozilla-foundation/common_voice_17_0 --hf-config ja --hf-split train --max-rows 200000

Hugging Face は `datasets` が入っていれば streaming で読む。無ければ、手元ログと
内蔵ドメイン語だけで辞書を作る。
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path.home() / ".koemo" / "native_corrections.json"

DOMAIN_TERMS = [
    "コエモ",
    "Koemo",
    "ライブ文字起こし",
    "文字起こし",
    "停止後10秒以内",
    "高速化テストです",
    "Windows純正音声認識",
    "精度確認",
    "これはコエモ高速化テストです",
    "ライブ文字起こしと停止後10秒以内の処理を確認しています",
    "Windows音声認識の確定結果がありません",
]

BASE_REGEX = [
    ["LIVE", "ライブ"],
    ["声\\s*mono", "コエモ"],
    ["声も", "コエモ"],
    ["こえも", "コエモ"],
    ["これも(?=高速化テスト)", "コエモ"],
    ["文字を腰", "文字起こし"],
    ["文字お腰", "文字起こし"],
    ["文字越し", "文字起こし"],
    ["文字腰", "文字起こし"],
    ["ライブ文字腰", "ライブ文字起こし"],
    ["制度が9人", "精度確認"],
    ["制度確認", "精度確認"],
    ["死後(?=\\d|[0-9０-９]|十)", "停止後"],
    ["高速化でストレスです", "高速化テストです"],
    ["高速化でストレス", "高速化テストです"],
]

DROP_IF_ALONE = ["っ", "の", "を", "8", "五"]


def clean_phrase(text):
    text = re.sub(r"\s+", "", (text or "").strip())
    text = re.sub(r"[^\w一-龯ぁ-んァ-ンー０-９0-9]+", "", text)
    if len(text) < 3 or len(text) > 28:
        return ""
    if re.fullmatch(r"[0-9０-９]+", text):
        return ""
    return text


def iter_koemo_texts():
    for path in (ROOT / "outputs" / "meetings").glob("summary_*.md"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            line = re.sub(r"^#+\s*", "", line).strip()
            line = re.sub(r"^\*\*[^*]+\*\*:\s*", "", line).strip()
            if line and not line.startswith(("-", "---")):
                yield line


def iter_hf_texts(dataset_name, config, split, text_column, max_rows):
    try:
        from datasets import Audio, load_dataset
    except ImportError as e:
        raise RuntimeError("Hugging Face datasets がありません: python -m pip install -r requirements-tools.txt") from e
    ds = load_dataset(dataset_name, config, split=split, streaming=True)
    try:
        features = getattr(ds, "features", {}) or {}
        for name, feature in features.items():
            if feature.__class__.__name__ == "Audio":
                ds = ds.cast_column(name, Audio(decode=False))
    except Exception:
        pass
    for i, row in enumerate(ds):
        if i >= max_rows:
            break
        text = row.get(text_column)
        if isinstance(text, str) and text.strip():
            yield text.strip()


def phrase_counter(texts):
    counter = Counter()
    for text in texts:
        for term in DOMAIN_TERMS:
            if term in text:
                counter[term] += 1
        for chunk in re.findall(r"[一-龯ぁ-んァ-ンA-Za-z0-9０-９ー]{4,24}", text):
            chunk = clean_phrase(chunk)
            if chunk:
                counter[chunk] += 1
    return counter


def build_rules(counter):
    rules = list(BASE_REGEX)
    # 公開/手元コーパスに多いKoemo関連語を残す。ここでは過剰補正を避け、既知の誤認識
    # パターンだけを増やす。汎用語を勝手に置換しない。
    if counter["ライブ文字起こし"] or counter["文字起こし"]:
        rules.extend([
            ["ライブ文字を腰", "ライブ文字起こし"],
            ["ライブ文字お腰", "ライブ文字起こし"],
            ["LIVE文字起こし", "ライブ文字起こし"],
        ])
    if counter["停止後10秒以内"]:
        rules.extend([
            ["停止50秒以内", "停止後10秒以内"],
            ["停止五十秒以内", "停止後10秒以内"],
            ["停止後十秒以内", "停止後10秒以内"],
        ])
    return rules


def build_grammar_phrases(counter, limit=300):
    phrases = []
    seen = set()
    for phrase in DOMAIN_TERMS:
        phrase = clean_phrase(phrase)
        if not phrase or phrase in seen:
            continue
        seen.add(phrase)
        phrases.append(phrase)
        if len(phrases) >= limit:
            break
    return phrases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--from-koemo-outputs", action="store_true")
    ap.add_argument("--hf-dataset", default="")
    ap.add_argument("--hf-config", default="ja")
    ap.add_argument("--hf-split", default="train")
    ap.add_argument("--hf-text-column", default="sentence")
    ap.add_argument("--max-rows", type=int, default=50000)
    args = ap.parse_args()

    texts = list(DOMAIN_TERMS)
    if args.from_koemo_outputs:
        texts.extend(iter_koemo_texts())
    if args.hf_dataset:
        texts.extend(iter_hf_texts(args.hf_dataset, args.hf_config, args.hf_split,
                                   args.hf_text_column, args.max_rows))
    counter = phrase_counter(texts)
    payload = {
        "source_note": "Generated from Koemo domain terms, optional Koemo outputs, and optional public HF text corpus.",
        "regex": build_rules(counter),
        "drop_if_alone": DROP_IF_ALONE,
        "grammar_phrases": build_grammar_phrases(counter),
        "top_phrases": [term for term, _count in counter.most_common(500)],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out} regex={len(payload['regex'])} phrases={len(payload['top_phrases'])}")


if __name__ == "__main__":
    main()
