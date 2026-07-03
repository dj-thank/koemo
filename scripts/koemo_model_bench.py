"""最新録音WAVで Whisper モデル別の速度と文字起こしを比較する。"""
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from koemo.transcribe import Transcriber
from koemo.gpu import enable_cuda_dlls, gpu_ok


def read_wav(path):
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    return sr, np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def main():
    enable_cuda_dlls()
    wavs = sorted((ROOT / "outputs" / "meetings").glob("recording_*_system.wav"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not wavs:
        raise SystemExit("no system wav")
    wav = wavs[0]
    sr, audio = read_wav(wav)
    results = []
    for model in ["small", "medium", "large-v3-turbo"]:
        transcriber = Transcriber(model, 2, 300, False)
        started = time.time()
        segs = transcriber.transcribe_segments(audio, language="ja")
        results.append({
            "model": model,
            "seconds": round(time.time() - started, 3),
            "text": " ".join(text for _st, _en, text in segs),
        })
    print(json.dumps({
        "wav": str(wav),
        "duration_sec": round(len(audio) / sr, 3),
        "gpu_ok": gpu_ok(),
        "results": results,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
