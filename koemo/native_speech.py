"""Windows純正音声認識のファイル再認識。"""
import json
import os
import subprocess
import sys
from pathlib import Path

from .native_correction import filter_native_texts, normalize_native_text


def _script_path():
    base = getattr(sys, "_MEIPASS", None)
    if base:
        path = Path(base) / "koemo" / "native_speech_file.ps1"
        if path.is_file():
            return str(path)
    return str(Path(__file__).resolve().parent / "native_speech_file.ps1")


def transcribe_wav_events(path, language="ja-JP", timeout_sec=45):
    """WAVをWindows純正 System.Speech で再認識し、信頼度つき断片を返す。"""
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", _script_path(), "-Path", str(path),
         "-Language", language, "-TimeoutSeconds", str(int(timeout_sec))],
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_sec + 10,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    events = []
    errors = []
    for raw in (proc.stdout or "").splitlines():
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "result" and (event.get("text") or "").strip():
            text = normalize_native_text((event.get("text") or "").strip())
            if text:
                try:
                    confidence = float(event.get("confidence") or 0.0)
                except Exception:
                    confidence = 0.0
                events.append({"text": text, "confidence": confidence})
        elif event.get("type") == "error":
            errors.append(event.get("error") or "Windows file speech failed")
    if proc.returncode != 0 and not events:
        detail = "; ".join(errors) or (proc.stderr or "").strip() or f"exit={proc.returncode}"
        raise RuntimeError(detail)
    filtered = []
    for text in filter_native_texts([e["text"] for e in events]):
        for event in events:
            if event["text"] == text:
                filtered.append(event)
                break
    return filtered


def transcribe_wav(path, language="ja-JP", timeout_sec=45):
    """WAVをWindows純正 System.Speech で再認識し、テキスト断片を返す。"""
    return [event["text"] for event in transcribe_wav_events(path, language, timeout_sec)]
