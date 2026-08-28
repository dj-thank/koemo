from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
from typing import Any


def _timecode(seconds: float | None, *, separator: str = ",") -> str:
    milliseconds = max(0, round(float(seconds or 0) * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def _variant_segments(document: dict[str, Any], variant: str) -> list[dict[str, Any]]:
    if variant == "normalized" and document.get("normalizedTranscript"):
        return document["normalizedTranscript"]["segments"]
    return document["observedTranscript"]["segments"]


def _variant_text(document: dict[str, Any], variant: str) -> str:
    if variant == "normalized" and document.get("normalizedTranscript"):
        return document["normalizedTranscript"]["text"]
    return document["observedTranscript"]["text"]


def render_srt(document: dict[str, Any], *, variant: str = "normalized") -> str:
    cues: list[str] = []
    for index, segment in enumerate(_variant_segments(document, variant), 1):
        speaker = segment.get("speaker")
        prefix = f"【{speaker}】" if speaker else ""
        text = f"{prefix}{segment.get('text', '').strip()}"
        cues.append(f"{index}\n{_timecode(segment.get('start'))} --> {_timecode(segment.get('end'))}\n{text}")
    return "\n\n".join(cues).rstrip() + "\n"


def render_vtt(document: dict[str, Any], *, variant: str = "normalized") -> str:
    cues = ["WEBVTT", ""]
    for index, segment in enumerate(_variant_segments(document, variant), 1):
        speaker = segment.get("speaker")
        prefix = f"【{speaker}】" if speaker else ""
        text = f"{prefix}{segment.get('text', '').strip()}"
        cues.extend([
            str(index),
            f"{_timecode(segment.get('start'), separator='.')} --> {_timecode(segment.get('end'), separator='.')}",
            text,
            "",
        ])
    return "\n".join(cues).rstrip() + "\n"


def render_markdown(document: dict[str, Any], *, variant: str = "normalized") -> str:
    source = document["source"]
    engine = document["engine"]
    lines = [
        f"# {source['name']} — 日本語文字起こし",
        "",
        f"- 入力SHA-256: `{source['sha256']}`",
        f"- モデル: `{engine.get('model')}`",
        f"- 言語: `{document['language'].get('code')}`",
        f"- 音声時間: `{document['duration'].get('seconds')}` 秒",
        f"- 観測文字列SHA-256: `{document['observedTranscript']['sha256']}`",
        "",
        "## 完成文字起こし",
        "",
        _variant_text(document, variant),
        "",
        "## タイムライン",
        "",
    ]
    for segment in _variant_segments(document, variant):
        speaker = f" **{segment['speaker']}**" if segment.get("speaker") else ""
        uncertainty = f" ⚠️ `{', '.join(segment['uncertaintyReasons'])}`" if segment.get("uncertaintyReasons") else ""
        lines.append(f"- `{_timecode(segment.get('start'), separator='.')}`{speaker} {segment.get('text', '').strip()}{uncertainty}")
    return "\n".join(lines).rstrip() + "\n"


def render_segments_tsv(document: dict[str, Any], *, variant: str = "normalized") -> str:
    output = io.StringIO()
    writer = csv.writer(output, delimiter="\t", lineterminator="\n")
    writer.writerow(["segment_id", "start_seconds", "end_seconds", "speaker", "text", "avg_logprob", "no_speech_prob", "uncertainty_reasons"])
    observed_by_id = {segment["id"]: segment for segment in document["observedTranscript"]["segments"]}
    for segment in _variant_segments(document, variant):
        observed = observed_by_id[segment["id"]]
        writer.writerow([
            segment["id"], segment.get("start"), segment.get("end"), segment.get("speaker") or "",
            segment.get("text", ""), observed.get("avgLogprob"), observed.get("noSpeechProb"),
            ",".join(observed.get("uncertaintyReasons", [])),
        ])
    return output.getvalue()


def render_words_jsonl(document: dict[str, Any]) -> str:
    lines: list[str] = []
    for segment in document["observedTranscript"]["segments"]:
        for word in segment.get("words", []):
            lines.append(json.dumps({"segmentId": segment["id"], **word}, ensure_ascii=False, separators=(",", ":")))
    return ("\n".join(lines) + "\n") if lines else ""


def atomic_write(path: str | Path, content: str, *, overwrite: bool = False) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {target}")
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, target)
    return target


def write_outputs(document: dict[str, Any], output_dir: str | Path, *, stem: str, formats: set[str], overwrite: bool = False, variant: str = "normalized") -> dict[str, str]:
    output_dir = Path(output_dir)
    outputs: dict[str, str] = {}
    renderers: dict[str, tuple[str, Any]] = {
        "json": ("transcript.json", lambda: json.dumps(document, ensure_ascii=False, indent=2) + "\n"),
        "txt": ("txt", lambda: _variant_text(document, variant).rstrip() + "\n"),
        "observed-txt": ("observed.txt", lambda: document["observedTranscript"]["text"].rstrip() + "\n"),
        "md": ("md", lambda: render_markdown(document, variant=variant)),
        "srt": ("srt", lambda: render_srt(document, variant=variant)),
        "vtt": ("vtt", lambda: render_vtt(document, variant=variant)),
        "tsv": ("segments.tsv", lambda: render_segments_tsv(document, variant=variant)),
        "words-jsonl": ("words.jsonl", lambda: render_words_jsonl(document)),
    }
    for name in sorted(formats):
        if name not in renderers:
            raise ValueError(f"unknown output format: {name}")
        suffix, renderer = renderers[name]
        path = output_dir / f"{stem}.{suffix}"
        atomic_write(path, renderer(), overwrite=overwrite)
        outputs[name] = str(path)
    return outputs
