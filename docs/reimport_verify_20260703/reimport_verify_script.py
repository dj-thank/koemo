"""Koemo 最終再import検証 — CT2 + Qwen2.5-3B 要約の実生成をログで確認する。

実行対象コードは本番と同一:
- koemo.gpu.enable_cuda_dlls() / gpu_ok()
- koemo.transcribe.Transcriber (faster-whisper large-v3-turbo)
- koemo.summarize.Summarizer -> koemo.backends.LocalCT2Backend (CTranslate2 + Qwen2.5-3B-Instruct-ct2-int8)

GPUがCode 43で無効な場合、gpu_ok()=Falseとなり自動でCPUにフォールバックする
(koemo/backends.py L55-61 の設計通り)。これはコードのバグではなく、
LocalCT2Backend.ensure_model() が意図した分岐。
"""
import json
import sys
import time
import wave
from pathlib import Path

ROOT = Path(r"C:\Users\rambo\RamboPC\DevHub\10_active\koemo")
sys.path.insert(0, str(ROOT))

from koemo import gpu
from koemo.transcribe import Transcriber
from koemo.summarize import Summarizer

WAV = Path(r"C:\Users\rambo\AppData\Local\Temp\claude\C--Users-rambo\44a8050c-4b2a-443e-ab5a-c7589a0e5c55\scratchpad\verify_tts.wav")


def read_wav(path):
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    import numpy as np
    return sr, np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def main():
    report = {"steps": []}

    # Step 1: GPU state (production entrypoint order: enable_cuda_dlls() then gpu_ok())
    gpu.enable_cuda_dlls()
    gpu_state = gpu.gpu_ok()
    report["gpu_ok"] = gpu_state
    report["steps"].append(f"gpu.gpu_ok() = {gpu_state} (Code43 GPU failure -> CPU fallback expected)")

    # Step 2: transcribe real audio through production Transcriber (faster-whisper/CT2 path)
    sr, audio = read_wav(WAV)
    duration = len(audio) / sr
    report["input_wav"] = str(WAV)
    report["input_duration_sec"] = round(duration, 3)

    transcriber = Transcriber("large-v3-turbo", cpu_threads=4, idle_sec=300, keep_warm=False)
    t0 = time.time()
    segs = transcriber.transcribe_segments(audio, language="ja")
    transcribe_sec = time.time() - t0
    transcript_text = " ".join(text for _st, _en, text in segs)
    report["transcribe_seconds"] = round(transcribe_sec, 3)
    report["transcript_text"] = transcript_text
    report["steps"].append(f"transcribe_segments(): {transcribe_sec:.3f}s, {len(segs)} segments")

    # Step 3: build a merge_rows-style labeled transcript (matches production _process shape)
    labeled_transcript = f"**相手**: {transcript_text}"

    # Step 4: summarize through production Summarizer -> LocalCT2Backend (CT2 + Qwen2.5-3B)
    summarizer = Summarizer(model_dir="", idle_sec=300, keep_warm=False, cfg={"summary_backend": "local"})
    progress_log = []
    t0 = time.time()
    title, body_md = summarizer.summarize(labeled_transcript, language="ja",
                                            on_progress=lambda msg: progress_log.append(msg))
    summarize_sec = time.time() - t0
    report["summarize_seconds"] = round(summarize_sec, 3)
    report["summary_backend_used"] = summarizer._backend.__class__.__name__
    report["summary_device"] = "cuda" if gpu_state else "cpu"
    report["progress_log"] = progress_log
    report["title"] = title
    report["summary_md"] = body_md
    report["steps"].append(f"summarize(): {summarize_sec:.3f}s via {summarizer._backend.__class__.__name__} "
                            f"(device={'cuda' if gpu_state else 'cpu'})")

    out_path = Path(r"C:\Users\rambo\AppData\Local\Temp\claude\C--Users-rambo\44a8050c-4b2a-443e-ab5a-c7589a0e5c55\scratchpad\reimport_verify_report.json")
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print("KOEMO REIMPORT VERIFICATION")
    print("=" * 70)
    for s in report["steps"]:
        print("- " + s)
    print("-" * 70)
    print("TRANSCRIPT:")
    print(transcript_text)
    print("-" * 70)
    print(f"TITLE: {title}")
    print("SUMMARY:")
    print(body_md)
    print("=" * 70)
    print(f"Full report: {out_path}")


if __name__ == "__main__":
    main()
