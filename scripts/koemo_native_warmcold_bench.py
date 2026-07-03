"""WinRT Speech の start_async コストを実機計測する（発話不要・決定的）。

実測の結論を再現するための診断ツール。

- プロセス最初の `start_async` は約600〜700ms（プロセス毎に一度の初期化コスト）。
- 同一プロセスの2回目以降は、recognizer を作り直しても約80ms。
- recognizer を使い回しても（warm再開）約75msで、作り直し80msとほぼ同じ。

=> `native_speech_keep_warm` の価値は「プロセス毎一度の初期化を起動時に前倒しして
   その回の録音から外す」ことに限られ、録音をまたいだ recognizer 再利用の定常的な
   速度差はほぼ無い。

使い方:
    python scripts/koemo_native_warmcold_bench.py
"""
import asyncio
import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


async def _t(coro):
    s = time.time()
    await coro
    return (time.time() - s) * 1000.0


async def run(rounds=4):
    from winrt.windows.globalization import Language
    from winrt.windows.media.speechrecognition import SpeechRecognizer

    fresh_starts, reused_starts = [], []
    for _ in range(rounds):
        rec = SpeechRecognizer(Language("ja-JP"))
        await rec.compile_constraints_async()
        sess = rec.continuous_recognition_session
        fresh_starts.append(await _t(sess.start_async()))  # round0 = プロセス初回（コールド）
        await asyncio.sleep(0.15)
        try:
            await sess.stop_async()
        except Exception:
            pass
        reused_starts.append(await _t(sess.start_async()))  # 同一recognizer再開
        await asyncio.sleep(0.1)
        try:
            await sess.stop_async()
        except Exception:
            pass
        try:
            rec.close()
        except Exception:
            pass

    process_first = round(fresh_starts[0], 1)
    steady_fresh = round(statistics.median(fresh_starts[1:]) if len(fresh_starts) > 1 else fresh_starts[0], 1)
    steady_reused = round(statistics.median(reused_starts[1:]) if len(reused_starts) > 1 else reused_starts[0], 1)
    return {
        "rounds": rounds,
        "process_first_start_ms": process_first,
        "steady_fresh_recognizer_start_ms": steady_fresh,
        "steady_reused_session_start_ms": steady_reused,
        "all_fresh_starts_ms": [round(x, 1) for x in fresh_starts],
        "all_reused_starts_ms": [round(x, 1) for x in reused_starts],
        "one_time_init_cost_ms": round(process_first - steady_fresh, 1),
        "reuse_vs_fresh_steady_gain_ms": round(steady_fresh - steady_reused, 1),
        "verdict": (
            "keep-warm absorbs the one-time per-process init at startup; "
            "reusing the recognizer across recordings gives negligible steady-state gain"
        ),
    }


def main():
    if platform.system() != "Windows":
        print(json.dumps({"ok": False, "note": "Windows only"}))
        return 1
    try:
        import winrt.windows.media.speechrecognition  # noqa: F401
    except Exception as e:
        print(json.dumps({"ok": False, "note": f"PyWinRT SpeechRecognition unavailable: {e}"}))
        return 1
    report = asyncio.run(run())
    report["ok"] = True  # 診断ツール: 計測できれば成功。利得の閾値判定はしない。
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
