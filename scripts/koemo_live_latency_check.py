"""Windows native live transcription latency check.

This is an environment-sensitive real-device check: Windows Speech listens to
the default microphone. The spoken TTS must be audible to that microphone.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundcard as sc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from koemo.live import NativeWindowsLiveTranscriber
from koemo.audio import DualRecorder
from koemo.config import MIC_LABEL
from koemo.live import MicActivityPreview


def start_mic_monitor(started, stop_flag, out):
    import threading

    def run():
        samples = []
        baseline_until = time.time() + 0.5
        baseline = 0.001
        try:
            mic = sc.default_microphone()
            with mic.recorder(samplerate=16000, channels=1, blocksize=400) as rec:
                while not stop_flag["stop"]:
                    chunk = rec.record(numframes=400)[:, 0].astype(np.float32, copy=False)
                    rms = float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) else 0.0
                    now = time.time()
                    if now < baseline_until:
                        samples.append(rms)
                        baseline = max(0.001, float(np.median(samples or [0.001])) * 4.0)
                        continue
                    armed_after = out.get("armed_after")
                    if armed_after is not None and now - started < armed_after:
                        continue
                    speech_threshold = max(0.01, baseline)
                    if out.get("mic_onset") is None and rms >= speech_threshold:
                        out["mic_onset"] = now - started
                        out["mic_rms"] = rms
                        out["threshold"] = speech_threshold
                        return
        except Exception as e:
            out["mic_error"] = str(e)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def start_speaker(text):
    escaped = text.replace("'", "''")
    script = f"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {{ $s.SelectVoice('Microsoft Haruka Desktop') }} catch {{}}
$s.Rate = -1
$s.Volume = 100
$OutputEncoding = [System.Text.Encoding]::UTF8
Write-Output 'KOEMO_TTS_READY'
$s.Speak('{escaped}')
$s.Dispose()
""".strip()
    return subprocess.Popen(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def main():
    events = []
    started = time.time()
    first_text_at = None
    first_activity_at = None
    mic_state = {"mic_onset": None, "armed_after": float("inf")}
    mic_stop = {"stop": False}
    mic_thread = start_mic_monitor(started, mic_stop, mic_state)

    def on_update(text):
        nonlocal first_text_at
        now = time.time() - started
        events.append({"seconds": round(now, 3), "text": text})
        if text and "ライブ文字起こし中" not in text and first_text_at is None:
            first_text_at = now

    transcriber = NativeWindowsLiveTranscriber(
        language="ja-JP",
        on_update=on_update,
        on_error=lambda msg: events.append({"seconds": round(time.time() - started, 3), "error": msg}),
    )
    recorder = DualRecorder({
        "sample_rate": 16000,
        "record_mic": True,
        "record_system": False,
        "save_dir": str(ROOT / ".codex_tmp" / "live_latency"),
        "mic_name": "",
        "speaker_name": "",
    })
    def on_activity(text):
        nonlocal first_activity_at
        if first_activity_at is None:
            first_activity_at = time.time() - started
        events.append({"seconds": round(time.time() - started, 3), "activity": text})

    activity = MicActivityPreview(recorder, on_update=on_activity)
    recorder.start()
    activity.start()
    # 計測ハーネスは認識開始（start_async + settle）を待ってからTTSを流す。
    # アプリ本体は wait=False でGUIをブロックしない（start() 既定）。
    transcriber.start(wait=True)
    try:
        preroll = float(__import__("os").environ.get("KOEMO_LATENCY_PREROLL_SEC", "0") or 0)
        if preroll > 0:
            time.sleep(preroll)
        speaker = start_speaker("テストテスト。ライブ文字起こし。停止後十秒以内。")
        assert speaker.stdout is not None
        tts_started = None
        ready_deadline = time.time() + 8.0
        while time.time() < ready_deadline:
            line = speaker.stdout.readline()
            if "KOEMO_TTS_READY" in line:
                tts_started = time.time()
                mic_state["armed_after"] = tts_started - started
                break
        if tts_started is None:
            raise RuntimeError("TTS did not start")
        deadline = time.time() + 8.0
        while time.time() < deadline and first_text_at is None:
            time.sleep(0.05)
        try:
            speaker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            speaker.kill()
    finally:
        mic_stop["stop"] = True
        mic_thread.join(timeout=1.0)
        activity.stop(timeout=0.2)
        try:
            recorder.stop()
        except Exception:
            pass
        transcriber.stop(timeout=1.0)

    tts_offset = tts_started - started
    mic_onset = mic_state.get("mic_onset")
    audio_basis = mic_onset if mic_onset is not None else tts_offset
    live_first = first_activity_at
    if first_text_at is not None:
        live_first = min(live_first, first_text_at) if live_first is not None else first_text_at
    report = {
        "ok": live_first is not None and live_first - audio_basis <= 1.0,
        "activity_under_1s_after_mic_onset": (
            first_activity_at is not None and mic_onset is not None and first_activity_at - mic_onset <= 1.0
        ),
        "windows_text_under_1s_after_mic_onset": (
            first_text_at is not None and mic_onset is not None and first_text_at - mic_onset <= 1.0
        ),
        "first_text_seconds_from_start": None if first_text_at is None else round(first_text_at, 3),
        "first_activity_seconds_from_start": None if first_activity_at is None else round(first_activity_at, 3),
        "tts_ready_seconds_from_start": round(tts_offset, 3),
        "mic_onset_seconds_from_start": None if mic_onset is None else round(mic_onset, 3),
        "mic_rms": mic_state.get("mic_rms"),
        "mic_threshold": mic_state.get("threshold"),
        "mic_error": mic_state.get("mic_error"),
        "first_text_seconds_after_tts": None if first_text_at is None else round(first_text_at - tts_offset, 3),
        "first_text_seconds_after_mic_onset": None if first_text_at is None or mic_onset is None else round(first_text_at - mic_onset, 3),
        "first_live_update_seconds_after_mic_onset": None if live_first is None or mic_onset is None else round(live_first - mic_onset, 3),
        "events": events[-10:],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
