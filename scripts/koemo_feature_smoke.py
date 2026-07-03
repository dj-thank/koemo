"""Koemo の実装機能をできるだけ実機に近く検証するスモークテスト。"""
import json
import os
import subprocess
import sys
import time
import wave
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / ".codex_tmp" / "feature_smoke"
TMP.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))


RESULTS = []


def check(name, func):
    started = time.time()
    try:
        detail = func()
        RESULTS.append({
            "name": name,
            "status": "pass",
            "seconds": round(time.time() - started, 3),
            "detail": detail or "",
        })
        print(f"[PASS] {name}")
    except Exception as e:
        RESULTS.append({
            "name": name,
            "status": "fail",
            "seconds": round(time.time() - started, 3),
            "detail": f"{type(e).__name__}: {e}",
        })
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def require(cond, msg):
    if not cond:
        raise AssertionError(msg)


def deps():
    import importlib.util
    mods = [
        "PySide6", "soundcard", "numpy", "keyboard",
        "faster_whisper", "ctranslate2", "transformers", "docx", "sherpa_onnx", "psutil",
        "httpx", "openai", "winrt.windows.media.speechrecognition",
    ]
    missing = [m for m in mods if importlib.util.find_spec(m) is None]
    require(not missing, f"missing: {missing}")
    return ", ".join(mods)


def config_defaults():
    import koemo.config as config_mod

    old_dir = config_mod.CONFIG_DIR
    old_file = config_mod.CONFIG_FILE
    config_mod.CONFIG_DIR = TMP / "config-defaults"
    config_mod.CONFIG_FILE = config_mod.CONFIG_DIR / "config.json"
    try:
        cfg = config_mod.load_config()
        for key in (
            "enable_live_transcription", "enable_meeting_detection",
            "summary_backend", "summary_sections", "summary_extra_instructions",
            "preload_transcriber", "fast_summary", "use_live_transcript_on_stop",
            "live_interval_sec", "live_window_sec", "live_stable_margin_sec",
            "live_min_audio_sec", "live_whisper_model", "preload_final_transcriber",
            "show_model_ready_status", "live_backend", "live_fallback_backend",
            "native_speech_language", "native_speech_startup_settle_sec", "final_channel_policy",
            "native_only_transcription", "enable_calendar_title_hint", "calendar_ics_path",
            "calendar_outlook_enabled", "calendar_title_lookback_min", "calendar_title_lookahead_min",
        ):
            require(key in config_mod.DEFAULT_CONFIG, f"DEFAULT_CONFIG missing {key}")
            require(key in cfg, f"load_config missing {key}")
        require(cfg["summary_backend"] in {"local", "ollama", "openai_compat"}, "bad backend")
        require(config_mod.DEFAULT_CONFIG["live_backend"] in {"auto", "native_windows", "whisper_rolling", "off"}, "bad live backend")
        require(config_mod.DEFAULT_CONFIG["live_fallback_backend"] == "whisper_rolling", "Whisper rolling fallback must be default")
        require(config_mod.DEFAULT_CONFIG["use_live_transcript_on_stop"] is False, "legacy live reuse must not be default")
        require(config_mod.DEFAULT_CONFIG["native_only_transcription"] is False, "Whisper final transcription must be default")
        require(config_mod.DEFAULT_CONFIG["whisper_model"] == "large-v3-turbo", "large-v3-turbo must be final default")
        require(float(config_mod.DEFAULT_CONFIG["native_speech_startup_settle_sec"]) >= 1.0, "native speech settle (display stabilization; non-blocking start) should stay >= 1.0")

        bad_dir = TMP / "bad-config"
        bad_dir.mkdir(exist_ok=True)
        config_mod.CONFIG_DIR = bad_dir
        config_mod.CONFIG_FILE = bad_dir / "config.json"
        config_mod.CONFIG_FILE.write_text("{bad json", encoding="utf-8")
        bad_cfg = config_mod.load_config()
        require("_config_load_error" in bad_cfg, "malformed config error not surfaced")
        require(not config_mod.CONFIG_FILE.exists(), "bad config was not moved aside")
        require(list(bad_dir.glob("config.invalid_*.json")), "bad config backup missing")
        config_mod.save_config({"hotkey": "ctrl+alt+r", "_config_load_error": "x"})
        saved = json.loads(config_mod.CONFIG_FILE.read_text(encoding="utf-8"))
        require("_config_load_error" not in saved and saved["hotkey"] == "ctrl+alt+r",
                "internal config keys were persisted")
        config_mod.CONFIG_FILE.write_text(json.dumps({
            "calendar_title_lookback_min": "abc",
            "calendar_title_lookahead_min": "9",
            "idle_unload_sec": -1,
            "enable_calendar_title_hint": "true",
            "unknown_key": "ignored",
        }), encoding="utf-8")
        coerced = config_mod.load_config()
        require(coerced["calendar_title_lookback_min"] == config_mod.DEFAULT_CONFIG["calendar_title_lookback_min"],
                "bad int config did not fall back to default")
        require(coerced["calendar_title_lookahead_min"] == 9, "numeric string was not coerced")
        require(coerced["idle_unload_sec"] == config_mod.DEFAULT_CONFIG["idle_unload_sec"],
                "negative numeric config did not fall back to default")
        require(coerced["enable_calendar_title_hint"] is True, "boolean string was not coerced")
        require("unknown_key" not in coerced, "unknown config key leaked")
        return "defaults merged + malformed config backup"
    finally:
        config_mod.CONFIG_DIR = old_dir
        config_mod.CONFIG_FILE = old_file


def no_exec_pattern():
    targets = [ROOT / "koemo.pyw"] + list((ROOT / "koemo").glob("*.py")) + [ROOT / "README.md", ROOT / "CODEX_TASKS.md"]
    hits = []
    for path in targets:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if ".exec(" in text:
            hits.append(str(path))
    require(not hits, f"forbidden pattern in {hits}")
    return "no .exec("


def audio_snapshot_live():
    import numpy as np
    from koemo.audio import DualRecorder
    from koemo.config import MIC_LABEL, SYS_LABEL
    from koemo.live import LiveEvent, LiveTranscriber, MicActivityPreview, NativeWindowsLiveTranscriber, format_live_events

    rec = DualRecorder({"sample_rate": 4})
    rec._live = {
        MIC_LABEL: [np.array([1, 2], dtype=np.float32), np.array([3, 4, 5], dtype=np.float32)],
        SYS_LABEL: [],
    }
    require(abs(rec.captured_seconds(MIC_LABEL) - 1.25) < 0.001, "captured_seconds")
    require(rec.snapshot(MIC_LABEL, max_seconds=0.75).tolist() == [3.0, 4.0, 5.0], "snapshot tail")

    class FakeRecorder:
        SR = 16000
        def __init__(self):
            self.sys_audio = np.zeros(self.SR * 12, dtype=np.float32)
            self.mic_audio = np.ones(self.SR * 12, dtype=np.float32) * 0.01
        def captured_seconds(self, label):
            return 12.0
        def snapshot(self, label, max_seconds=None):
            samples = int(max_seconds * self.SR)
            audio = self.sys_audio if label == SYS_LABEL else self.mic_audio
            return audio[-samples:]

    class FakeTranscriber:
        def transcribe_segments(self, audio, language="ja", on_progress=None, **kwargs):
            dur = len(audio) / 16000
            return [(0.0, 1.0, "最初"), (dur - 2.0, dur - 1.0, "最新")]

    updates = []
    lt = LiveTranscriber(FakeRecorder(), FakeTranscriber(), on_update=updates.append)
    lt._transcribe_once()
    require(updates and "あなた: 最新" in updates[-1], "live mic fallback")
    rendered = format_live_events([
        LiveEvent(0, 1, MIC_LABEL, "速報", "provisional"),
        LiveEvent(1, 2, MIC_LABEL, "確定", "final"),
    ])
    require("速報 ..." in rendered and "確定" in rendered, rendered)
    events = lt.events()
    require(events and events[-1].kind == "final" and events[-1].label == MIC_LABEL, "whisper live event contract")
    activity_updates = []
    activity = MicActivityPreview(rec, on_update=activity_updates.append, silence_rms=0.001)
    activity.start()
    time.sleep(0.2)
    activity.stop()
    require(activity_updates and "あなた: ..." in activity_updates[-1], "mic activity preview missing")
    require(activity.events() and activity.events()[0].kind == "provisional", "activity event contract")
    nw_contract = NativeWindowsLiveTranscriber()
    nw_contract._on_result_text("確定")
    nw_contract._on_hypothesis_text("速報")
    native_events = nw_contract.events()
    require(any(ev.kind == "final" for ev in native_events), "native final event contract")
    require(any(ev.kind == "provisional" for ev in native_events), "native provisional event contract")
    if not NativeWindowsLiveTranscriber.available():
        nw = NativeWindowsLiveTranscriber()
        try:
            nw.start()
        except RuntimeError as e:
            require("Windows" in str(e) or "PyWinRT" in str(e), str(e))
        else:
            nw.stop()
    return updates[-1]


def live_fallback_contract():
    from koemo import app as appmod
    from koemo.app import KoemoApp

    class Emitter:
        def __init__(self, out):
            self.out = out
        def emit(self, value=None):
            self.out.append(value)

    class FakeSig:
        def __init__(self):
            self.live_values = []
            self.toast_values = []
            self.live = Emitter(self.live_values)
            self.toast = Emitter(self.toast_values)

    class FailingNative:
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            raise RuntimeError("speech disabled")

    class FallbackLive:
        def __init__(self, *args, **kwargs):
            self.args = args
        def start(self):
            pass
        def stop(self, timeout=1.0):
            pass
        def final_rows(self):
            return []

    old_native = appmod.NativeWindowsLiveTranscriber
    old_live = appmod.LiveTranscriber
    try:
        appmod.NativeWindowsLiveTranscriber = FailingNative
        appmod.LiveTranscriber = FallbackLive
        obj = KoemoApp.__new__(KoemoApp)
        obj.cfg = {"live_fallback_backend": "whisper_rolling", "summary_language": "ja"}
        obj.recorder = object()
        obj.live_model = object()
        obj.sig = FakeSig()
        backend = KoemoApp._make_live_transcriber(obj, "native_windows")
        require(isinstance(backend, FallbackLive), "native failure did not fallback to whisper rolling")
        require(any("Whisper速報に切替" in str(v) for v in obj.sig.live_values), "fallback status missing")
    finally:
        appmod.NativeWindowsLiveTranscriber = old_native
        appmod.LiveTranscriber = old_live
    return "native unavailable falls back to whisper_rolling with visible status"


def live_backend_auto_resolves_by_gpu_contract():
    """live_backend="auto" を実機で解決する契約: GPUありなら日本語精度の高い
    whisper_rolling、無ければ軽い native_windows。明示指定はそのまま尊重する。"""
    import koemo.gpu as gpumod
    from koemo.app import KoemoApp

    obj = KoemoApp.__new__(KoemoApp)
    orig = gpumod.gpu_ok
    try:
        gpumod.gpu_ok = lambda: True
        obj.cfg = {"live_backend": "auto"}
        require(obj._resolve_live_backend() == "whisper_rolling", "auto+GPU must resolve to whisper_rolling")
        gpumod.gpu_ok = lambda: False
        require(obj._resolve_live_backend() == "native_windows", "auto+noGPU must resolve to native_windows")
        for explicit in ("native_windows", "whisper_rolling", "off"):
            obj.cfg = {"live_backend": explicit}
            require(obj._resolve_live_backend() == explicit, f"explicit {explicit} must be preserved")
    finally:
        gpumod.gpu_ok = orig
    return "auto -> whisper_rolling(GPU)/native_windows(CPU); explicit backends preserved"


def live_start_nonblocking_contract():
    import platform
    from koemo.live import NativeWindowsLiveTranscriber

    if platform.system() != "Windows":
        return "skipped (non-Windows)"

    # 既定(wait=False)はGUIスレッドをブロックしない。settle相当の遅延後に
    # ready を立てる遅い worker を擬似し、start() が即returnすることを確認する。
    nw = NativeWindowsLiveTranscriber(startup_settle_sec=1.0)

    def slow_main():
        time.sleep(1.0)
        nw._ready.set()

    nw._thread_main = slow_main
    t0 = time.time()
    nw.start()
    nonblocking_elapsed = time.time() - t0
    nw.stop(timeout=2.0)
    require(nonblocking_elapsed < 0.3,
            f"start() blocked the caller for {nonblocking_elapsed:.3f}s (should be non-blocking)")

    # wait=True（計測ハーネス）は ready まで待つ。
    nw2 = NativeWindowsLiveTranscriber(startup_settle_sec=0.0)

    def slow_main2():
        time.sleep(0.5)
        nw2._ready.set()

    nw2._thread_main = slow_main2
    t1 = time.time()
    nw2.start(wait=True)
    blocking_elapsed = time.time() - t1
    nw2.stop(timeout=2.0)
    require(blocking_elapsed >= 0.4,
            f"start(wait=True) should block until ready, blocked={blocking_elapsed:.3f}s")
    return f"start() {nonblocking_elapsed:.3f}s non-blocking, start(wait=True) {blocking_elapsed:.3f}s blocking"


def live_async_failure_falls_back_contract():
    import platform
    from koemo.live import NativeWindowsLiveTranscriber

    if platform.system() != "Windows":
        return "skipped (non-Windows)"

    errs, unavail = [], []
    nw = NativeWindowsLiveTranscriber(
        on_error=lambda m: errs.append(m),
        on_unavailable=lambda m: unavail.append(m),
    )

    async def boom_winrt():
        raise RuntimeError("winrt unavailable")

    def boom_bridge():
        raise RuntimeError("bridge unavailable")

    # WinRT と System.Speech bridge の両方が ready前に失敗する状況を擬似する。
    nw._run_winrt_async = boom_winrt
    nw._run_bridge = boom_bridge
    nw.start()  # 非ブロック
    deadline = time.time() + 3.0
    while time.time() < deadline and not unavail:
        time.sleep(0.02)
    nw.stop(timeout=1.0)
    require(unavail, "pre-ready WinRT+bridge failure did not trigger on_unavailable (fallback hook)")
    require(errs, "pre-ready failure did not trigger on_error")
    return "pre-ready native failure triggers on_unavailable so the app can fall back"


def native_state_detected_event_contract():
    import types
    from koemo.config import MIC_LABEL
    from koemo.live import NativeWindowsLiveTranscriber

    updates = []
    nw = NativeWindowsLiveTranscriber(on_update=updates.append)
    nw._on_state(types.SimpleNamespace(name="SPEECH_DETECTED"))
    require(updates and updates[-1] == f"{MIC_LABEL}: ...",
            f"engine speech-detected did not move live UI: {updates}")

    # 実テキストが既にある場合、発話検知で上書きしない。
    nw._on_hypothesis_text("実テキスト")
    before = list(updates)
    nw._on_state(types.SimpleNamespace(name="SOUND_STARTED"))
    require(updates == before, "speech-detected clobbered an existing hypothesis")
    return "engine speech-detected moves live UI before first hypothesis text"


def live_activity_prearms_before_native_contract():
    from koemo import app as appmod
    from koemo.app import KoemoApp

    calls = []

    class FakeLiveWindow:
        def update_text(self, text):
            calls.append(("window", text))
        def show_at_corner(self):
            calls.append(("window", "show"))

    class FakeActivity:
        def __init__(self, *args, **kwargs):
            calls.append(("activity", "init"))
        def start(self):
            calls.append(("activity", "start"))
        def stop(self, timeout=0.2):
            calls.append(("activity", "stop"))

    class FakeNative:
        def __init__(self, *args, **kwargs):
            calls.append(("native", "init"))
        def start(self):
            calls.append(("native", "start"))
        def stop(self, timeout=1.0):
            calls.append(("native", "stop"))
        def final_rows(self):
            return []

    class Emitter:
        def emit(self, value=None):
            calls.append(("emit", value))

    class FakeSig:
        live = Emitter()
        toast = Emitter()

    old_window = appmod.LiveWindow
    old_activity = appmod.MicActivityPreview
    old_native = appmod.NativeWindowsLiveTranscriber
    try:
        appmod.LiveWindow = FakeLiveWindow
        appmod.MicActivityPreview = FakeActivity
        appmod.NativeWindowsLiveTranscriber = FakeNative
        obj = KoemoApp.__new__(KoemoApp)
        obj.cfg = {"enable_live_transcription": True, "live_backend": "native_windows"}
        obj.recorder = object()
        obj.sig = FakeSig()
        obj._live_window = None
        obj._live_activity = None
        obj._live_transcriber = None
        KoemoApp._start_live(obj)
        first_activity_start = calls.index(("activity", "start"))
        first_native_start = calls.index(("native", "start"))
        require(first_activity_start < first_native_start, f"activity did not prearm: {calls}")
    finally:
        appmod.LiveWindow = old_window
        appmod.MicActivityPreview = old_activity
        appmod.NativeWindowsLiveTranscriber = old_native
    return "mic activity preview starts before native speech startup wait"


def native_live_respects_record_mic_contract():
    """record_mic=false ではOSの既定マイクを開く native live を起動しない。"""
    from koemo import app as appmod
    from koemo.app import KoemoApp

    calls = []

    class FakeLiveWindow:
        def update_text(self, text):
            calls.append(("window", text))
        def show_at_corner(self):
            calls.append(("window", "show"))

    class ForbiddenActivity:
        def __init__(self, *args, **kwargs):
            raise AssertionError("MicActivityPreview must not start when record_mic=false")

    class ForbiddenNative:
        def __init__(self, *args, **kwargs):
            raise AssertionError("NativeWindowsLiveTranscriber must not start when record_mic=false")

    class FakeRolling:
        def __init__(self, *args, **kwargs):
            calls.append(("rolling", "init"))
        def start(self):
            calls.append(("rolling", "start"))
        def stop(self, timeout=1.0):
            calls.append(("rolling", "stop"))
        def final_rows(self):
            return []

    class Emitter:
        def emit(self, value=None):
            calls.append(("emit", value))

    class FakeSig:
        live = Emitter()
        toast = Emitter()

    old_window = appmod.LiveWindow
    old_activity = appmod.MicActivityPreview
    old_native = appmod.NativeWindowsLiveTranscriber
    old_rolling = appmod.LiveTranscriber
    try:
        appmod.LiveWindow = FakeLiveWindow
        appmod.MicActivityPreview = ForbiddenActivity
        appmod.NativeWindowsLiveTranscriber = ForbiddenNative
        appmod.LiveTranscriber = FakeRolling
        obj = KoemoApp.__new__(KoemoApp)
        obj.cfg = {
            "enable_live_transcription": True,
            "live_backend": "native_windows",
            "live_fallback_backend": "whisper_rolling",
            "record_mic": False,
            "record_system": True,
            "summary_language": "ja",
        }
        obj.recorder = object()
        obj.live_model = object()
        obj.sig = FakeSig()
        obj._live_window = None
        obj._live_activity = None
        obj._live_transcriber = None
        KoemoApp._start_live(obj)
        require(("rolling", "start") in calls, f"whisper rolling fallback did not start: {calls}")
        require(not any(kind == "activity" or kind == "native" for kind, _value in calls), calls)
    finally:
        appmod.LiveWindow = old_window
        appmod.MicActivityPreview = old_activity
        appmod.NativeWindowsLiveTranscriber = old_native
        appmod.LiveTranscriber = old_rolling
    return "record_mic=false uses recorded-channel rolling live instead of opening native mic"


def final_ignores_live_rows_contract():
    import numpy as np
    from koemo.app import KoemoApp
    from koemo.config import MIC_LABEL

    class Emitter:
        def __init__(self, out):
            self.out = out
        def emit(self, value=None):
            self.out.append(value)

    class FakeSig:
        def __init__(self):
            self.toasts = []
            self.errors = []
            self.results = []
            self.toast = Emitter(self.toasts)
            self.toast_close = Emitter(self.toasts)
            self.error = Emitter(self.errors)
            self.results = Emitter(self.results)

    class FakeRecorder:
        def stop(self):
            return {
                "ts": "20990101_000000",
                "duration": 2,
                "channels": {MIC_LABEL: np.ones(32000, dtype=np.float32) * 0.01},
                "write_errors": {},
            }

    class FakeTranscriber:
        def transcribe_segments(self, audio, language="ja", on_progress=None):
            return [(0.0, 1.0, "正式テキスト")]

    class FakeSummarizer:
        def fast_summarize(self, transcript, language="ja"):
            return "正式テキスト", "## 要旨\n正式テキスト"

    out_dir = TMP / "final_ignores_live"
    out_dir.mkdir(exist_ok=True)
    doc = out_dir / "summary_20990101_000000.md"
    if doc.exists():
        doc.unlink()

    obj = KoemoApp.__new__(KoemoApp)
    obj.cfg = {
        "save_dir": str(out_dir),
        "summary_language": "ja",
        "native_only_transcription": False,
        "use_live_transcript_on_stop": False,
        "fast_summary": True,
        "enable_diarization": False,
        "final_channel_policy": "auto_dedupe",
    }
    obj.recorder = FakeRecorder()
    obj.transcriber = FakeTranscriber()
    obj.summarizer = FakeSummarizer()
    obj._last_live_rows = [(0.0, MIC_LABEL, "ライブ誤認識")]
    obj.sig = FakeSig()
    obj.processing = True
    obj._add_library = lambda *args, **kwargs: None
    KoemoApp._process(obj)
    text = doc.read_text(encoding="utf-8")
    require("正式テキスト" in text, "final transcript missing")
    require("ライブ誤認識" not in text, "live transcript leaked into final transcript")
    return str(doc)


def final_drops_saturated_mic_echo_contract():
    """停止後(final)経路: スピーカー音をマイクが飽和エコーで拾い、システム
    ループバックだけが正しい内容を持つケースを決定的に再現する。マイクが
    システムより大音量でも、クリップ飽和を検知してクリーンなシステムchを
    優先し、Whisperが幻聴するエコーchを捨てることを検証する。"""
    import numpy as np
    from koemo.app import KoemoApp
    from koemo.config import MIC_LABEL, SYS_LABEL

    sr = 16000
    t = np.arange(sr * 4) / sr
    # システム(相手): クリーンなループバック = 本当の内容。クリップしない。
    sys_audio = (0.25 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    # マイク(あなた): スピーカー音を過大入力で拾った飽和エコー。大音量で激しくクリップ。
    mic_audio = np.clip(2.2 * np.sin(2 * np.pi * 180 * t + 0.4), -1.0, 1.0).astype(np.float32)

    mic_clip = float(np.mean(np.abs(mic_audio) >= 0.98))
    sys_clip = float(np.mean(np.abs(sys_audio) >= 0.98))
    mic_rms = float(np.sqrt(np.mean(mic_audio ** 2)))
    sys_rms = float(np.sqrt(np.mean(sys_audio ** 2)))
    require(mic_clip > 0.2 and sys_clip == 0.0, f"fixture invalid mic_clip={mic_clip} sys_clip={sys_clip}")
    require(mic_rms >= sys_rms * 1.8, f"fixture: mic must be louder (mic={mic_rms:.3f} sys={sys_rms:.3f})")

    obj = KoemoApp.__new__(KoemoApp)
    obj.cfg = {"final_channel_policy": "auto_dedupe"}
    picked = [label for label, _audio in obj._active_final_channels({MIC_LABEL: mic_audio, SYS_LABEL: sys_audio})]
    require(picked == [SYS_LABEL],
            f"saturated mic echo not dropped: louder mic kept over clean system loopback, picked={picked}")

    # 逆ケース: マイクがクリーンで大きいなら従来どおりマイク採用を壊さない。
    clean_loud_mic = (0.5 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    picked2 = [label for label, _audio in obj._active_final_channels({MIC_LABEL: clean_loud_mic, SYS_LABEL: sys_audio})]
    require(MIC_LABEL in picked2, f"clean louder mic wrongly dropped by echo gate: picked={picked2}")

    # 通常の同時発話/会議音声は、文字起こし前の音量差だけでは片chを捨てない。
    normal_mic = (0.10 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    normal_sys = (0.08 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
    picked_normal = [label for label, _audio in obj._active_final_channels({MIC_LABEL: normal_mic, SYS_LABEL: normal_sys})]
    require(picked_normal == [MIC_LABEL, SYS_LABEL],
            f"normal active mic/system channels should both be kept: picked={picked_normal}")

    # GPU Whisper は、AEC後に低音量で残ったスピーカー漏れも別発話として
    # 文字にしてしまうことがある。system が圧倒的に強く mic が小さい時は
    # クリーンな system のみを正式 transcript にする。
    low_leak_mic = (0.041 * np.sin(2 * np.pi * 180 * t + 0.2)).astype(np.float32)
    loud_system = (0.36 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    picked_leak = [label for label, _audio in obj._active_final_channels({MIC_LABEL: low_leak_mic, SYS_LABEL: loud_system})]
    require(picked_leak == [SYS_LABEL],
            f"low-level mic system leak not dropped: picked={picked_leak}")

    # all_active は明示ポリシーなのでエコーでも全ch維持する（明示指定の尊重）。
    obj.cfg = {"final_channel_policy": "all_active"}
    picked3 = [label for label, _audio in obj._active_final_channels({MIC_LABEL: low_leak_mic, SYS_LABEL: loud_system})]
    require(MIC_LABEL in picked3 and SYS_LABEL in picked3,
            f"all_active must keep every active channel even on echo: picked={picked3}")
    return f"mic_clip={mic_clip*100:.0f}% saturated echo dropped; low-level leak dropped; all_active keeps both"


def final_audio_finite_samples_contract():
    """NaN/inf が混ざっても音量・クリップ判定は有限サンプルだけで行う。"""
    import numpy as np
    from koemo.app import KoemoApp

    noisy = np.array([np.nan, np.inf, -np.inf, 0.5, -0.5], dtype=np.float32)
    clipped = np.array([np.nan, np.inf, -np.inf, 1.0, -1.0, 0.0, 0.1], dtype=np.float32)
    require(abs(KoemoApp._rms(noisy) - 0.5) < 0.0001,
            f"finite-only rms failed: {KoemoApp._rms(noisy)}")
    require(abs(KoemoApp._clip_fraction(clipped) - 0.5) < 0.0001,
            f"finite-only clip fraction failed: {KoemoApp._clip_fraction(clipped)}")
    return "NaN/inf ignored; finite samples keep rms=0.5 and clip_fraction=0.5"


def final_filters_common_whisper_hallucination_contract():
    from koemo.app import KoemoApp
    from koemo.config import MIC_LABEL, SYS_LABEL

    obj = KoemoApp.__new__(KoemoApp)
    rows = [
        (0.0, MIC_LABEL, "ご視聴ありがとうございました"),
        (0.0, SYS_LABEL, "これはコエモ高速化テストです"),
        (1.0, SYS_LABEL, "ライブ文字起こしと停止後10秒以内の処理を確認しています"),
    ]
    filtered = KoemoApp._filter_final_rows(obj, rows)
    require(all("ご視聴ありがとうございました" not in text for _st, _label, text in filtered),
            f"common hallucination row was not filtered: {filtered}")
    require(len(filtered) == 2 and all(label == SYS_LABEL for _st, label, _text in filtered),
            f"expected surviving system rows only: {filtered}")
    single_channel = KoemoApp._filter_final_rows(obj, [(0.0, MIC_LABEL, "ご視聴ありがとうございました")])
    require(single_channel, "single-channel content must not be dropped entirely")
    return "common Whisper outro hallucination is dropped only when another channel has content"


def aec_batched_wiener_contract():
    """AEC は長尺相当でも全フレームSTFTを一括確保せず、エコーを抑える。"""
    import numpy as np
    from koemo import audio as audio_mod

    sr = 16000
    t = np.arange(sr * 8) / sr
    ref = (0.35 * np.sin(2 * np.pi * 440 * t)
           + 0.20 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
    near = (0.08 * np.sin(2 * np.pi * 220 * t + 0.3)).astype(np.float32)
    echo = np.concatenate([np.zeros(300, dtype=np.float32), ref[:-300]]) * 0.8
    mic = (echo + near).astype(np.float32)
    cleaned = audio_mod.cancel_echo(mic, ref)

    def corr(a, b):
        n = min(len(a), len(b))
        a = a[:n] - a[:n].mean()
        b = b[:n] - b[:n].mean()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    require(abs(corr(cleaned, ref)) < 0.05,
            f"cleaned signal still tracks loopback echo: corr={corr(cleaned, ref):.3f}")
    require(corr(cleaned, near) > 0.85,
            f"near-end speech was not preserved: corr={corr(cleaned, near):.3f}")

    frame = 8192
    hop = 2048
    batch_limit = 256
    n = frame + hop * (batch_limit + 17)
    t = np.arange(n) / sr
    ref_long = (0.20 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
    mic_long = (0.5 * ref_long).astype(np.float32)
    old_rfft = audio_mod.np.fft.rfft
    seen_batches = []

    def spy_rfft(a, *args, **kwargs):
        shape = getattr(a, "shape", ())
        if len(shape) == 2 and shape[1] == frame:
            seen_batches.append(shape[0])
            require(shape[0] <= batch_limit,
                    f"AEC rfft saw {shape[0]} frames at once; expected <= {batch_limit}")
        return old_rfft(a, *args, **kwargs)

    try:
        audio_mod.np.fft.rfft = spy_rfft
        audio_mod.cancel_echo(mic_long, ref_long, frame=frame, hop=hop)
    finally:
        audio_mod.np.fft.rfft = old_rfft

    require(seen_batches and max(seen_batches) <= batch_limit,
            f"batch spy did not observe bounded STFT batches: {seen_batches[:5]}")
    return f"echo suppressed; max_stft_batch={max(seen_batches)}"


def recording_capture_errors_surface_contract():
    """片側録音スレッドが失敗しても、残ったchと警告を呼び出し側へ返す。"""
    import numpy as np
    from datetime import datetime
    from koemo.audio import DualRecorder
    from koemo.config import MIC_LABEL, SYS_LABEL

    rec_dir = TMP / "capture_error_surface"
    rec_dir.mkdir(exist_ok=True)
    rec = DualRecorder({"sample_rate": 16000, "save_dir": str(rec_dir), "enable_aec": False})
    rec._threads = []
    rec._start_ts = datetime(2099, 1, 1, 0, 0, 0)
    rec._buf = {
        MIC_LABEL: np.ones(1600, dtype=np.float32) * 0.01,
        SYS_LABEL + "_err": "loopback device failed",
    }
    data = rec.stop()
    require(data is not None and MIC_LABEL in data["channels"], "surviving mic channel was lost")
    require(data["capture_errors"].get(SYS_LABEL) == "loopback device failed",
            f"capture error not surfaced: {data.get('capture_errors')}")
    require((rec_dir / "recording_20990101_000000_mic.wav").is_file(), "surviving wav not saved")
    return "capture_errors returned with surviving channel"


def recording_save_dir_failure_nonfatal_contract():
    """WAV保存先の作成に失敗しても、録音音声そのものは処理へ返す。"""
    import numpy as np
    from datetime import datetime
    from koemo.audio import DualRecorder
    from koemo.config import MIC_LABEL

    bad_save_dir = TMP / "save_dir_is_a_file"
    bad_save_dir.write_text("not a directory", encoding="utf-8")
    rec = DualRecorder({"sample_rate": 16000, "save_dir": str(bad_save_dir), "enable_aec": False})
    rec._threads = []
    rec._start_ts = datetime(2099, 1, 1, 0, 0, 2)
    rec._buf = {MIC_LABEL: np.ones(1600, dtype=np.float32) * 0.01}

    data = rec.stop()
    require(data is not None and MIC_LABEL in data["channels"], "save_dir failure lost recorded audio")
    require(data["files"] == {}, f"files should be empty when save_dir cannot be created: {data['files']}")
    require(data["write_errors"].get("_save_dir"), f"save_dir creation failure not recorded: {data['write_errors']}")
    return "save_dir creation failure is surfaced without blocking transcript processing"


def recording_all_capture_errors_surface_contract():
    """全ch失敗でも None で黙らず、capture_errors を返してアプリが通知できる。"""
    from datetime import datetime
    from koemo.audio import DualRecorder
    from koemo.app import KoemoApp
    from koemo.config import MIC_LABEL, SYS_LABEL

    rec = DualRecorder({"sample_rate": 16000, "save_dir": str(TMP), "enable_aec": False})
    rec._threads = []
    rec._start_ts = datetime(2099, 1, 1, 0, 0, 1)
    rec._buf = {
        MIC_LABEL + "_err": "mic failed",
        SYS_LABEL + "_err": "loopback failed",
    }
    data = rec.stop()
    require(data is not None and data["channels"] == {}, "all-channel failure returned None or fake audio")
    require(data["capture_errors"].get(MIC_LABEL) == "mic failed", "mic error missing")
    require(data["capture_errors"].get(SYS_LABEL) == "loopback failed", "system error missing")

    class Emitter:
        def __init__(self, out):
            self.out = out
        def emit(self, value=None):
            self.out.append(value)

    class FakeSig:
        def __init__(self):
            self.toasts = []
            self.errors = []
            self.result_payloads = []
            self.toast = Emitter(self.toasts)
            self.toast_close = Emitter(self.toasts)
            self.error = Emitter(self.errors)
            self.results = Emitter(self.result_payloads)

    class FakeRecorder:
        def stop(self):
            return data

    obj = KoemoApp.__new__(KoemoApp)
    obj.cfg = {"save_dir": str(TMP), "summary_language": "ja"}
    obj.recorder = FakeRecorder()
    obj._last_live_rows = []
    obj.sig = FakeSig()
    obj.processing = True
    obj._pending_cfg = None
    KoemoApp._process(obj)
    require(obj.sig.errors and "音声取得に失敗" in obj.sig.errors[0],
            f"all-channel capture failure not surfaced: {obj.sig.errors}")
    return "all-channel capture_errors surfaced as user-visible error"


def recording_no_channel_config_rejected_contract():
    """mic/system両方OFFの不可能な録音設定は開始せず、ユーザーに通知する。"""
    from koemo.app import KoemoApp

    calls = []

    class Emitter:
        def __init__(self, name):
            self.name = name
        def emit(self, value=None):
            calls.append((self.name, value))

    class FakeSig:
        toast = Emitter("toast")
        error = Emitter("error")

    class ForbiddenRecorder:
        def start(self):
            raise AssertionError("recorder.start must not be called when no channels are enabled")

    obj = KoemoApp.__new__(KoemoApp)
    obj.cfg = {"record_mic": False, "record_system": False}
    obj.sig = FakeSig()
    obj.recorder = ForbiddenRecorder()
    obj.recording = False
    obj._final_model_state = "ready"
    KoemoApp._start(obj)
    require(obj.recording is False, "recording flag was set for no-channel config")
    require(any(name == "error" and "録音対象がありません" in str(value) for name, value in calls),
            f"no-channel error was not emitted: {calls}")
    return "mic/system both disabled is rejected before recording starts"


def recorder_spools_audio_and_bounds_live_contract():
    """録音本体はディスクへspoolし、ライブ用リングだけを小さく保持する。"""
    import numpy as np
    from koemo.audio import DualRecorder
    from koemo.config import MIC_LABEL

    class FakeReader:
        def __init__(self, rec):
            self.rec = rec
            self.calls = 0
        def record(self, numframes=1600):
            self.calls += 1
            if self.calls > 12:
                self.rec.recording = False
            return np.ones((numframes, 1), dtype=np.float32) * (self.calls / 100.0)

    class FakeCtx:
        def __init__(self, reader):
            self.reader = reader
        def __enter__(self):
            return self.reader
        def __exit__(self, *_args):
            return False

    class FakeDev:
        def __init__(self, rec):
            self.rec = rec
        def recorder(self, **_kwargs):
            return FakeCtx(FakeReader(self.rec))

    rec_dir = TMP / "spooled_capture"
    rec_dir.mkdir(exist_ok=True)
    rec = DualRecorder({"sample_rate": 16000, "save_dir": str(rec_dir), "enable_aec": False})
    rec.recording = True
    rec._buf = {}
    rec._live = {MIC_LABEL: []}
    rec._live_samples = {MIC_LABEL: 0}
    rec._captured_samples = {MIC_LABEL: 0}
    rec._live_max_samples = 1600 * 3
    rec._capture(lambda: FakeDev(rec), MIC_LABEL)
    item = rec._buf.get(MIC_LABEL)
    require(isinstance(item, dict) and Path(item["path"]).is_file(), "capture did not spool to a temp file")
    require(item["samples"] > rec._live_max_samples, f"spooled capture was not longer than live ring: {item}")
    require(abs(rec.captured_seconds(MIC_LABEL) - (item["samples"] / 16000.0)) < 0.01,
            f"captured_seconds lost total length: {rec.captured_seconds(MIC_LABEL)}")
    require(len(rec.snapshot(MIC_LABEL)) <= 1600 * 3,
            f"live snapshot exceeded ring bound: {len(rec.snapshot(MIC_LABEL))}")
    Path(item["path"]).unlink(missing_ok=True)
    return "capture spooled to disk; live ring bounded while total samples kept"


def recorder_hung_thread_reports_temp_contract():
    """join timeoutした録音スレッドは警告化し、既知spoolをcleanup対象に残す。"""
    import numpy as np
    from datetime import datetime
    from koemo.audio import DualRecorder
    from koemo.app import KoemoApp
    from koemo.config import MIC_LABEL, SYS_LABEL

    out_dir = TMP / "hung_recorder_spool"
    tmp_dir = out_dir / ".koemo_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    mic_spool = tmp_dir / "capture_mic_hung.f32"
    sys_spool = tmp_dir / "capture_system_ok.f32"
    np.ones(16000, dtype=np.float32).tofile(mic_spool)
    (np.ones(16000, dtype=np.float32) * 0.02).tofile(sys_spool)

    class HungThread:
        def join(self, timeout=None):
            self.join_timeout = timeout
        def is_alive(self):
            return True

    rec = DualRecorder({"save_dir": str(out_dir), "sample_rate": 16000, "enable_aec": False})
    rec.recording = True
    rec._start_ts = datetime(2099, 1, 1, 5, 0, 0)
    hung = HungThread()
    rec._threads = [hung]
    rec._thread_labels = {hung: MIC_LABEL}
    rec._buf = {SYS_LABEL: {"path": str(sys_spool), "samples": 16000}}
    with rec._temp_lock:
        rec._temp_files = [str(mic_spool), str(sys_spool)]
    got = rec.stop()
    require(got and SYS_LABEL in got["channels"], "surviving channel was not returned")
    require(MIC_LABEL in got.get("capture_errors", {}), f"hung capture error missing: {got}")
    require(str(mic_spool) in got.get("temp_files", []), "hung mic spool was not kept for cleanup")
    require(str(sys_spool) in got.get("temp_files", []), "normal spool was not kept for cleanup")
    KoemoApp._cleanup_recording_temp_files(got)
    require(not mic_spool.exists() and not sys_spool.exists(), "known temp files were not cleaned")
    return "hung capture thread is reported and known spool files are cleaned"


def settings_save_during_recording_contract():
    """録音/処理中の設定保存では recorder を差し替えず、次回録音から反映する。"""
    import koemo.app as appmod
    from koemo.app import KoemoApp

    class FakeSummarizer:
        def reload(self, *args, **kwargs):
            return None

    class FakeRecorder:
        def __init__(self, cfg):
            self.cfg = cfg

    class Emitter:
        def __init__(self):
            self.values = []
        def emit(self, value=None):
            self.values.append(value)

    class FakeSig:
        def __init__(self):
            self.toast = Emitter()

    old_dual_recorder = appmod.DualRecorder
    created = []

    def fake_dual_recorder(cfg):
        rec = FakeRecorder(cfg)
        created.append(rec)
        return rec

    obj = KoemoApp.__new__(KoemoApp)
    obj.cfg = {"native_only_transcription": True, "save_dir": "old"}
    obj.recorder = object()
    original_recorder = obj.recorder
    obj.transcriber = object()
    obj.live_model = object()
    obj.summarizer = FakeSummarizer()
    obj.sig = FakeSig()
    obj._pending_cfg = None
    obj._register_hotkey = lambda: None
    obj._start_meeting_watcher = lambda: None
    obj._preload_transcriber = lambda: None

    try:
        appmod.DualRecorder = fake_dual_recorder
        new_cfg = {"native_only_transcription": True, "save_dir": "x"}
        obj.recording = True
        obj.processing = False
        KoemoApp._on_cfg_saved(obj, new_cfg)
        require(obj.recorder is original_recorder and not created,
                "recorder was replaced while recording")
        require(obj.cfg["save_dir"] == "old" and obj._pending_cfg == new_cfg,
                "recording save was applied instead of deferred")

        newer_cfg = {"native_only_transcription": True, "save_dir": "y"}
        obj.recording = False
        obj.processing = True
        KoemoApp._on_cfg_saved(obj, newer_cfg)
        require(obj.recorder is original_recorder and not created,
                "recorder was replaced while processing")
        require(obj.cfg["save_dir"] == "old" and obj._pending_cfg == newer_cfg,
                "processing save was applied instead of deferred")

        obj.processing = False
        KoemoApp._apply_pending_cfg_if_idle(obj)
        require(created and obj.recorder is created[-1],
                "deferred idle apply did not prepare a new recorder")
        require(obj.cfg["save_dir"] == "y", "deferred config was not applied after processing")
    finally:
        appmod.DualRecorder = old_dual_recorder
    return "recording/processing saves are deferred; idle apply rebuilds recorder"


def full_summary_releases_whisper_models_contract():
    """full LLM要約前にWhisperモデルを解放し、3モデル同時VRAM常駐を避ける。"""
    import numpy as np
    from koemo.app import KoemoApp
    from koemo.config import MIC_LABEL

    class Emitter:
        def __init__(self, out):
            self.out = out
        def emit(self, value=None):
            self.out.append(value)

    class FakeSig:
        def __init__(self):
            self.toasts = []
            self.errors = []
            self.result_payloads = []
            self.toast = Emitter(self.toasts)
            self.toast_close = Emitter(self.toasts)
            self.error = Emitter(self.errors)
            self.results = Emitter(self.result_payloads)

    class FakeRecorder:
        def stop(self):
            return {
                "ts": "20990101_010000",
                "duration": 1,
                "channels": {MIC_LABEL: np.ones(16000, dtype=np.float32) * 0.02},
                "files": {},
                "write_errors": {},
                "capture_errors": {},
            }

    class FakeTranscriber:
        def __init__(self, name):
            self.name = name
            self.unloaded = False
        def transcribe_segments(self, audio, language="ja", on_progress=None):
            require(not self.unloaded, f"{self.name} unloaded before final transcription")
            return [(0.0, 1.0, "正式テキスト")]
        def unload(self):
            self.unloaded = True
            return True

    out_dir = TMP / "full_summary_release"
    out_dir.mkdir(exist_ok=True)
    final_model = FakeTranscriber("final")
    live_model = FakeTranscriber("live")

    class FakeSummarizer:
        def summarize(self, transcript, language="ja", on_progress=None):
            require(final_model.unloaded, "final Whisper model was not unloaded before LLM summary")
            require(live_model.unloaded, "live Whisper model was not unloaded before LLM summary")
            return "正式テキスト", "## 要旨\n正式テキスト"

    obj = KoemoApp.__new__(KoemoApp)
    obj.cfg = {
        "save_dir": str(out_dir),
        "summary_language": "ja",
        "native_only_transcription": False,
        "use_live_transcript_on_stop": False,
        "fast_summary": False,
        "enable_diarization": False,
        "final_channel_policy": "auto_dedupe",
        "keep_warm": False,
    }
    obj.recorder = FakeRecorder()
    obj.transcriber = final_model
    obj.live_model = live_model
    obj.summarizer = FakeSummarizer()
    obj._last_live_rows = []
    obj.sig = FakeSig()
    obj.processing = True
    obj._add_library = lambda *args, **kwargs: None
    KoemoApp._process(obj)

    require(not obj.sig.errors, f"process emitted errors: {obj.sig.errors}")
    require((out_dir / "summary_20990101_010000.md").is_file(), "summary not written")
    return "full summary path releases live/final Whisper before loading LLM"


def import_full_summary_releases_whisper_models_contract():
    """音声取込のfull LLM要約でもWhisperを先に解放する。"""
    from koemo.app import KoemoApp

    class Emitter:
        def __init__(self, out):
            self.out = out
        def emit(self, value=None):
            self.out.append(value)

    class FakeSig:
        def __init__(self):
            self.toasts = []
            self.errors = []
            self.result_payloads = []
            self.toast = Emitter(self.toasts)
            self.toast_close = Emitter(self.toasts)
            self.error = Emitter(self.errors)
            self.results = Emitter(self.result_payloads)

    class FakeTranscriber:
        def __init__(self, name):
            self.name = name
            self.unloaded = False
        def transcribe_segments(self, path, language="ja", on_progress=None):
            require(not self.unloaded, f"{self.name} unloaded before import transcription")
            return [(0.0, 1.0, "取込テキスト")]
        def unload(self):
            self.unloaded = True
            return True

    out_dir = TMP / "import_full_summary_release"
    out_dir.mkdir(exist_ok=True)
    src = out_dir / "input.wav"
    src.write_bytes(b"placeholder")
    final_model = FakeTranscriber("final")
    live_model = FakeTranscriber("live")

    class FakeSummarizer:
        def summarize(self, transcript, language="ja", on_progress=None):
            require(final_model.unloaded, "final Whisper model was not unloaded before import LLM summary")
            require(live_model.unloaded, "live Whisper model was not unloaded before import LLM summary")
            return "取込テキスト", "## 要旨\n取込テキスト"

    obj = KoemoApp.__new__(KoemoApp)
    obj.cfg = {
        "save_dir": str(out_dir),
        "summary_language": "ja",
        "native_only_transcription": False,
        "fast_summary": False,
        "keep_warm": False,
    }
    obj.transcriber = final_model
    obj.live_model = live_model
    obj.summarizer = FakeSummarizer()
    obj.sig = FakeSig()
    obj.processing = True
    obj._pending_cfg = None
    obj._add_library = lambda *args, **kwargs: None
    KoemoApp._process_file(obj, str(src))

    require(not obj.sig.errors, f"import emitted errors: {obj.sig.errors}")
    require(obj.sig.result_payloads, "import did not emit results")
    return "import full summary path releases live/final Whisper before loading LLM"


def import_save_dir_failure_falls_back_contract():
    """音声取込でも保存先作成失敗時はLOCALAPPDATA側へfallbackし、結果を失わない。"""
    import os
    from koemo.app import KoemoApp

    class Emitter:
        def __init__(self, out):
            self.out = out
        def emit(self, value=None):
            self.out.append(value)

    class FakeSig:
        def __init__(self):
            self.toasts = []
            self.errors = []
            self.result_payloads = []
            self.toast = Emitter(self.toasts)
            self.toast_close = Emitter(self.toasts)
            self.error = Emitter(self.errors)
            self.results = Emitter(self.result_payloads)

    class FakeTranscriber:
        def transcribe_segments(self, path, language="ja", on_progress=None):
            return [(0.0, 1.0, "取込保存fallbackテキスト")]

    class FakeSummarizer:
        def fast_summarize(self, transcript, language="ja"):
            return "取込保存fallback", "## 要旨\n取込保存fallback"

    out_dir = TMP / "import_save_dir_failure"
    out_dir.mkdir(exist_ok=True)
    bad_save_dir = out_dir / "save_dir_is_file"
    bad_save_dir.write_text("not a directory", encoding="utf-8")
    src = out_dir / "input.wav"
    src.write_bytes(b"placeholder")
    localappdata = out_dir / "localappdata"
    old_localappdata = os.environ.get("LOCALAPPDATA")

    obj = KoemoApp.__new__(KoemoApp)
    obj.cfg = {
        "save_dir": str(bad_save_dir),
        "summary_language": "ja",
        "native_only_transcription": False,
        "fast_summary": True,
    }
    obj.transcriber = FakeTranscriber()
    obj.summarizer = FakeSummarizer()
    obj.sig = FakeSig()
    obj.processing = True
    obj._pending_cfg = None
    obj._add_library = lambda *args, **kwargs: None

    try:
        os.environ["LOCALAPPDATA"] = str(localappdata)
        KoemoApp._process_file(obj, str(src))
    finally:
        if old_localappdata is None:
            os.environ.pop("LOCALAPPDATA", None)
        else:
            os.environ["LOCALAPPDATA"] = old_localappdata

    require(not obj.sig.errors, f"import emitted errors: {obj.sig.errors}")
    require(obj.sig.result_payloads, "import did not emit results")
    payload = obj.sig.result_payloads[-1]
    require(str(payload["dir"]).startswith(str(localappdata / "Koemo" / "Recordings")),
            f"import did not fall back to LOCALAPPDATA: {payload['dir']}")
    require("保存警告" in payload["summary"], "fallback save warning missing from result summary")
    docs = list((localappdata / "Koemo" / "Recordings").glob("import_*.md"))
    require(docs and "保存警告" in docs[-1].read_text(encoding="utf-8"), "fallback import file missing warning")
    return "import save_dir failure falls back with warning"


def process_cleans_spool_files_contract():
    """通常処理後に録音spoolファイルを残さない。"""
    import numpy as np
    from koemo.app import KoemoApp
    from koemo.config import MIC_LABEL

    class Emitter:
        def __init__(self, out):
            self.out = out
        def emit(self, value=None):
            self.out.append(value)

    class FakeSig:
        def __init__(self):
            self.toasts = []
            self.errors = []
            self.result_payloads = []
            self.toast = Emitter(self.toasts)
            self.toast_close = Emitter(self.toasts)
            self.error = Emitter(self.errors)
            self.results = Emitter(self.result_payloads)

    out_dir = TMP / "process_spool_cleanup"
    tmp_dir = out_dir / ".koemo_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    spool = tmp_dir / "capture_mic_test.f32"
    samples = (np.ones(16000, dtype=np.float32) * 0.02)
    samples.tofile(spool)

    class FakeRecorder:
        def stop(self):
            mm = np.memmap(str(spool), dtype=np.float32, mode="r", shape=(16000,))
            return {
                "ts": "20990101_030000",
                "duration": 1,
                "channels": {MIC_LABEL: mm},
                "files": {},
                "write_errors": {},
                "capture_errors": {},
                "temp_files": [str(spool)],
            }

    class FakeTranscriber:
        def transcribe_segments(self, audio, language="ja", on_progress=None):
            require(len(audio) == 16000, "spooled memmap not passed to transcriber")
            return [(0.0, 1.0, "spool cleanup text")]

    class FakeSummarizer:
        def fast_summarize(self, transcript, language="ja"):
            return "spool cleanup", "## 要旨\nspool cleanup"

    obj = KoemoApp.__new__(KoemoApp)
    obj.cfg = {
        "save_dir": str(out_dir),
        "summary_language": "ja",
        "native_only_transcription": False,
        "use_live_transcript_on_stop": False,
        "fast_summary": True,
        "enable_diarization": False,
        "final_channel_policy": "auto_dedupe",
    }
    obj.recorder = FakeRecorder()
    obj.transcriber = FakeTranscriber()
    obj.summarizer = FakeSummarizer()
    obj._last_live_rows = []
    obj.sig = FakeSig()
    obj.processing = True
    obj._pending_cfg = None
    obj._add_library = lambda *args, **kwargs: None
    KoemoApp._process(obj)

    require(not obj.sig.errors, f"process emitted errors: {obj.sig.errors}")
    require(not spool.exists(), f"spool file was left behind: {spool}")
    leftovers = list(tmp_dir.glob("capture_*.f32"))
    require(not leftovers, f"spool leftovers remain: {leftovers}")
    return "recording spool files are removed after processing"


def process_surfaces_spool_cleanup_failure_contract():
    """削除に失敗したspoolパスは保持し、保存summaryに警告として出す。"""
    import numpy as np
    from koemo.app import KoemoApp
    from koemo.config import MIC_LABEL

    class Emitter:
        def __init__(self, out):
            self.out = out
        def emit(self, value=None):
            self.out.append(value)

    class FakeSig:
        def __init__(self):
            self.toasts = []
            self.errors = []
            self.result_payloads = []
            self.toast = Emitter(self.toasts)
            self.toast_close = Emitter(self.toasts)
            self.error = Emitter(self.errors)
            self.results = Emitter(self.result_payloads)

    out_dir = TMP / "process_spool_cleanup_failure"
    tmp_dir = out_dir / ".koemo_tmp"
    locked = tmp_dir / "capture_locked.f32"
    if locked.exists() and locked.is_dir():
        locked.rmdir()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    locked.mkdir()

    class FakeRecorder:
        def __init__(self):
            self.last = None
        def stop(self):
            self.last = {
                "ts": "20990101_040000",
                "duration": 1,
                "channels": {MIC_LABEL: np.ones(16000, dtype=np.float32) * 0.02},
                "files": {},
                "write_errors": {},
                "capture_errors": {},
                "temp_files": [str(locked)],
            }
            return self.last

    class FakeTranscriber:
        def transcribe_segments(self, audio, language="ja", on_progress=None):
            return [(0.0, 1.0, "cleanup warning text")]

    class FakeSummarizer:
        def fast_summarize(self, transcript, language="ja"):
            return "cleanup warning", "## 要旨\ncleanup warning"

    recorder = FakeRecorder()
    obj = KoemoApp.__new__(KoemoApp)
    obj.cfg = {
        "save_dir": str(out_dir),
        "summary_language": "ja",
        "native_only_transcription": False,
        "use_live_transcript_on_stop": False,
        "fast_summary": True,
        "enable_diarization": False,
        "final_channel_policy": "auto_dedupe",
    }
    obj.recorder = recorder
    obj.transcriber = FakeTranscriber()
    obj.summarizer = FakeSummarizer()
    obj._last_live_rows = []
    obj.sig = FakeSig()
    obj.processing = True
    obj._pending_cfg = None
    obj._add_library = lambda *args, **kwargs: None
    try:
        KoemoApp._process(obj)
        require(not obj.sig.errors, f"process emitted errors: {obj.sig.errors}")
        require(recorder.last is not None, "recorder was not used")
        require(str(locked) in recorder.last.get("temp_files", []), "failed cleanup path was not preserved")
        require(recorder.last.get("temp_cleanup_errors"), "cleanup failure was not recorded")
        summary = out_dir / "summary_20990101_040000.md"
        text = summary.read_text(encoding="utf-8")
        require("## 一時ファイル警告" in text, "cleanup warning not written to summary")
        require("capture_locked.f32" in text, "failed temp path not included in warning")
    finally:
        if locked.exists() and locked.is_dir():
            locked.rmdir()
    return "spool cleanup failures are preserved and surfaced"


def preload_state_contract():
    from koemo.app import KoemoApp

    class Emitter:
        def __init__(self, out):
            self.out = out
        def emit(self, value=None):
            self.out.append(value)

    class FakeSig:
        def __init__(self):
            self.toasts = []
            self.toast = Emitter(self.toasts)
            self.toast_close = Emitter(self.toasts)

    class FakeTranscriber:
        def __init__(self):
            self.warmed = False
        def warmup(self):
            self.warmed = True

    obj = KoemoApp.__new__(KoemoApp)
    obj.cfg = {
        "native_only_transcription": False,
        "preload_transcriber": True,
        "preload_final_transcriber": True,
        "show_model_ready_status": True,
        "live_backend": "whisper_rolling",
        "record_mic": True,
    }
    obj.transcriber = FakeTranscriber()
    obj.live_model = FakeTranscriber()
    obj.sig = FakeSig()
    obj._final_model_state = "idle"
    KoemoApp._preload_transcriber(obj)
    deadline = time.time() + 5.0
    while time.time() < deadline and not (
        obj._final_model_state == "ready" and obj.transcriber.warmed and obj.live_model.warmed
    ):
        time.sleep(0.05)
    require(obj._final_model_state == "ready", f"state={obj._final_model_state}")
    require(obj.transcriber.warmed and obj.live_model.warmed, "models not warmed")
    require(any("高精度モデル準備完了" in str(t) for t in obj.sig.toasts), "ready toast missing")

    obj2 = KoemoApp.__new__(KoemoApp)
    obj2.cfg = {
        "native_only_transcription": False,
        "preload_transcriber": True,
        "preload_final_transcriber": True,
        "show_model_ready_status": False,
        "live_backend": "native_windows",
        "record_mic": True,
    }
    obj2.transcriber = FakeTranscriber()
    obj2.live_model = FakeTranscriber()
    obj2.sig = FakeSig()
    obj2._final_model_state = "idle"
    KoemoApp._preload_transcriber(obj2)
    deadline = time.time() + 5.0
    while time.time() < deadline and obj2._final_model_state != "ready":
        time.sleep(0.05)
    require(obj2._final_model_state == "ready", f"state2={obj2._final_model_state}")
    require(obj2.transcriber.warmed, "final model not warmed")
    require(not obj2.live_model.warmed, "native_windows should not preload unused live Whisper model")
    return "final preload avoids unused live Whisper model"


def real_short_recording():
    from koemo.audio import DualRecorder
    from koemo.app import KoemoApp
    from koemo.config import MIC_LABEL, SYS_LABEL

    rec_dir = TMP / "recordings"
    rec_dir.mkdir(exist_ok=True)
    cfg = {
        "sample_rate": 16000,
        "record_mic": True,
        "record_system": True,
        "enable_aec": True,
        "save_dir": str(rec_dir),
        "mic_name": "",
        "speaker_name": "",
    }
    rec = DualRecorder(cfg)
    rec.start()
    time.sleep(3)
    data = rec.stop()
    try:
        require(data is not None, "recorder returned None")
        require(data["channels"], "no channels")
        labels = list(data["channels"])
        if MIC_LABEL in data["channels"]:
            require((rec_dir / f"recording_{data['ts']}_mic.wav").is_file(), "mic wav missing")
        if SYS_LABEL in data["channels"]:
            require((rec_dir / f"recording_{data['ts']}_system.wav").is_file(), "system wav missing")
        return f"channels={labels} duration={data['duration']}"
    finally:
        KoemoApp._cleanup_recording_temp_files(data)


def summarizer_templates_chat_backends():
    from koemo.backends import OllamaBackend, OpenAICompatBackend
    from koemo.summarize import Summarizer

    class FakeBackend:
        def generate(self, system, user, max_tokens=1024):
            if "タイトル" in user:
                return "リリース会議"
            if "質問:" in user:
                return "決定事項はリリース実施です。"
            return "## 概要\nリリース方針を確認しました。\n## TODO\n- 山田さんが確認します。"
        def unload(self):
            return False

    s = Summarizer(cfg={"summary_sections": ["概要", "TODO"], "summary_extra_instructions": "簡潔に"})
    s._make_backend = lambda: FakeBackend()
    title, body = s.summarize("**相手**: 明日リリースします。山田さんが確認します。")
    require(title == "リリース会議", title)
    require("## 概要" in body and "## TODO" in body, body)
    fast_title, fast_body = s.fast_summarize("**相手**: 明日リリースします。山田さんが確認します。")
    require(fast_title and "## 概要" in fast_body and "## TODO" in fast_body, fast_body)
    require("リリース実施" in s.chat("決定事項は？", "**相手**: 明日リリースします。"), "chat")
    long_transcript = "\n".join([
        "**相手**: " + ("前半の雑談です。" * 900),
        "**相手**: 中間の契約番号は K-42 です。ここだけに答えがあります。",
        "**相手**: " + ("終盤の雑談です。" * 900),
    ])
    ctx = Summarizer._select_chat_context("契約番号は？", long_transcript,
                                          chunk_chars=2000, max_context_chars=7000)
    require("K-42" in ctx, "long chat context dropped relevant middle chunk")
    try:
        OpenAICompatBackend().generate("s", "u")
    except RuntimeError as e:
        require("base_url" in str(e), "openai error")
    else:
        raise AssertionError("OpenAICompatBackend should require config")
    require(OllamaBackend().model, "ollama model default")
    return title


def library_roundtrip():
    from koemo import library
    db = TMP / "library-smoke.db"
    if db.exists():
        db.unlink()
    old = library.DB_FILE
    library.DB_FILE = db
    try:
        mid = library.add("リリース会議", "## 要旨\n確認", "**相手**: リリース", TMP, 61, "20260530_120000")
        require(library.get(mid)["title"] == "リリース会議", "get")
        require(library.recent(), "recent")
        require(library.search("リリース"), "search")
    finally:
        library.DB_FILE = old
    db.unlink()
    require(not db.exists(), "db lock remains")
    return "sqlite add/get/search/unlock"


def calendar_title_hint_contract():
    from datetime import datetime, timezone
    from koemo.calendar_hint import apply_title_hint, current_title, load_ics_events

    ics = TMP / "calendar-title.ics"
    ics.write_text("""BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
UID:koemo-test
DTSTART:20260531T010000Z
DTEND:20260531T020000Z
SUMMARY:朝会\\, Koemo進捗
END:VEVENT
END:VCALENDAR
""", encoding="utf-8")
    now = datetime(2026, 5, 31, 1, 30, tzinfo=timezone.utc)
    events = load_ics_events(ics, now)
    require(len(events) == 1 and events[0].title == "朝会, Koemo進捗", events)
    cfg = {
        "enable_calendar_title_hint": True,
        "calendar_ics_path": str(ics),
        "calendar_outlook_enabled": False,
        "calendar_title_lookback_min": 15,
        "calendar_title_lookahead_min": 10,
    }
    require(current_title(cfg, now) == "朝会, Koemo進捗", "current event title not selected")
    require(apply_title_hint(cfg, "生成タイトル", now) == "朝会, Koemo進捗", "hint did not override generated title")
    cfg["enable_calendar_title_hint"] = False
    require(apply_title_hint(cfg, "生成タイトル", now) == "生成タイトル", "disabled hint changed title")

    ics_overlap = TMP / "calendar-title-overlap.ics"
    ics_overlap.write_text("""BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
UID:koemo-active
DTSTART:20260531T010000Z
DTEND:20260531T021000Z
SUMMARY:進行中の会議
END:VEVENT
BEGIN:VEVENT
UID:koemo-upcoming
DTSTART:20260531T015500Z
DTEND:20260531T023000Z
SUMMARY:次の会議
END:VEVENT
END:VCALENDAR
""", encoding="utf-8")
    cfg["enable_calendar_title_hint"] = True
    cfg["calendar_ics_path"] = str(ics_overlap)
    cfg["calendar_title_lookback_min"] = 60
    cfg["calendar_title_lookahead_min"] = 10
    require(current_title(cfg, datetime(2026, 5, 31, 1, 50, tzinfo=timezone.utc)) == "進行中の会議",
            "active event must beat a nearer upcoming event")

    ics_tz = TMP / "calendar-title-tzid.ics"
    ics_tz.write_text("""BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
UID:koemo-tz-test
DTSTART;TZID=Asia/Tokyo:20260531T100000
DTEND;TZID=Asia/Tokyo:20260531T110000
SUMMARY:東京朝会
END:VEVENT
END:VCALENDAR
""", encoding="utf-8")
    cfg["enable_calendar_title_hint"] = True
    cfg["calendar_ics_path"] = str(ics_tz)
    require(current_title(cfg, now) == "東京朝会", "TZID event title not selected")
    ics_win_tz = TMP / "calendar-title-windows-tzid.ics"
    ics_win_tz.write_text("""BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
UID:koemo-windows-tz-test
DTSTART;TZID=Tokyo Standard Time:20260531T100000
DTEND;TZID=Tokyo Standard Time:20260531T110000
SUMMARY:Windows TZ朝会
END:VEVENT
END:VCALENDAR
""", encoding="utf-8")
    cfg["calendar_ics_path"] = str(ics_win_tz)
    require(current_title(cfg, now) == "Windows TZ朝会", "Windows TZID event title not selected")
    cfg["calendar_title_lookback_min"] = "abc"
    cfg["calendar_title_lookahead_min"] = "abc"
    require(current_title(cfg, now) == "Windows TZ朝会", "bad calendar window config should fall back")
    cfg["calendar_title_lookback_min"] = -1
    cfg["calendar_title_lookahead_min"] = -1
    require(current_title(cfg, now) == "Windows TZ朝会", "negative calendar window config should fall back")
    return "ICS current event title can become the meeting title"


def diarize_models():
    from koemo import diarize
    require(diarize.available(), "diarization models not available")
    rows = diarize.assign_speakers([(0.0, 1.0, "a"), (1.0, 2.0, "b")],
                                   [(0.0, 1.0, 0), (1.0, 2.0, 1)], "相手")
    require(rows[0][1] == "相手1" and rows[1][1] == "相手2", rows)
    return "models available"


def exports_and_ui():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from koemo.app import make_icon
    from koemo.export import export_docx, export_markdown, export_pdf
    from koemo.ui_chat import ChatWindow
    from koemo.ui_library import LibraryWindow
    from koemo.ui_live import LiveWindow
    from koemo.ui_results import ResultsWindow
    import koemo.ui_settings as settings_mod
    from koemo.ui_settings import SettingsDialog

    require(not make_icon(False).isNull(), "icon")
    md = "# テスト\n\n本文"
    export_markdown(md, TMP / "smoke.md")
    export_docx(md, TMP / "smoke.docx")
    export_pdf(md, TMP / "smoke.pdf")
    for name in ("smoke.md", "smoke.docx", "smoke.pdf"):
        require((TMP / name).is_file(), name)

    lw = LiveWindow(); lw.update_text("相手: テスト"); lw.close()
    rw = ResultsWindow("テスト", "## 要旨\nOK", "**相手**: OK", TMP, 3, chat_func=lambda q, h: "OK"); rw.close()
    cw = ChatWindow("テスト", lambda q, h: "OK"); cw.close()
    lib = LibraryWindow(chat_factory=lambda tr: (lambda q, h: "OK")); lib.close()
    old_save_config = settings_mod.save_config
    saved = []
    try:
        settings_mod.save_config = lambda cfg: saved.append(dict(cfg))
        cfg = {"save_dir": str(TMP), "final_channel_policy": "auto_dedupe"}
        sd = SettingsDialog(cfg, lambda c: saved.append({"callback": c.get("final_channel_policy")}))
        require(hasattr(sd, "co_final_channel_policy"), "final channel policy UI missing")
        sd._select(sd.co_final_channel_policy, "all_active")
        sd._save()
        require(cfg["final_channel_policy"] == "auto_dedupe", "settings dialog mutated live cfg before callback")
        require(any(item.get("final_channel_policy") == "all_active" for item in saved if isinstance(item, dict)),
                "final channel policy not saved")
        require(saved, "settings save was not invoked")
    finally:
        settings_mod.save_config = old_save_config
        try:
            sd.close()
        except Exception:
            pass
    app.processEvents()
    return "exports + widgets"


def meeting_detection():
    from koemo.detect import MEETING_PROCESSES, MeetingWatcher
    require("zoom.exe" in MEETING_PROCESSES, "zoom missing")
    require("slack.exe" not in MEETING_PROCESSES, "plain Slack launch should not be a meeting")
    seen = []
    mw = MeetingWatcher(seen.append, interval_sec=0.1)
    mw.start()
    time.sleep(0.2)
    mw.stop()
    return "watcher start/stop"


def packaging_files():
    exe = ROOT / "dist" / "Koemo" / "Koemo.exe"
    internal = ROOT / "dist" / "Koemo" / "_internal"
    bridge = ROOT / "dist" / "Koemo" / "_internal" / "koemo" / "native_speech_bridge.ps1"
    file_bridge = ROOT / "dist" / "Koemo" / "_internal" / "koemo" / "native_speech_file.ps1"
    corrections = ROOT / "dist" / "Koemo" / "_internal" / "koemo" / "data" / "native_corrections.json"
    require((ROOT / "koemo.spec").is_file(), "spec missing")
    require(exe.is_file(), "exe missing")
    require(bridge.is_file(), "native speech bridge missing from EXE bundle")
    require(file_bridge.is_file(), "native speech file recognizer missing from EXE bundle")
    require(corrections.is_file(), "native corrections missing from EXE bundle")
    require((ROOT / "assets" / "koemo.png").is_file(), "png missing")
    require((ROOT / "assets" / "koemo.ico").is_file(), "ico missing")
    bundled_bloat = [
        "datasets", "pandas", "pyarrow", "fastapi", "uvicorn", "starlette",
        "aiohttp", "tornado", "mypy", "PIL", "_tcl_data",
    ]
    found = [name for name in bundled_bloat if (internal / name).exists()]
    require(not found, f"unexpected non-runtime packages bundled: {found}")
    bloat_metadata = [
        "datasets-*dist-info", "pandas-*dist-info", "pyarrow-*dist-info",
        "fastapi-*dist-info", "uvicorn-*dist-info", "starlette-*dist-info",
        "aiohttp-*dist-info", "tornado-*dist-info", "Pillow-*dist-info",
        "mypy-*dist-info", "pytest_*dist-info", "torch-*dist-info", "torchaudio-*dist-info",
        "torchvision-*dist-info", "scipy-*dist-info", "scikit_learn-*dist-info",
    ]
    found_meta = []
    for pattern in bloat_metadata:
        found_meta.extend(p.name for p in internal.glob(pattern))
    require(not found_meta, f"unexpected non-runtime package metadata bundled: {found_meta}")
    pyz_toc = ROOT / "build" / "koemo" / "PYZ-00.toc"
    if pyz_toc.is_file():
        pyz_text = pyz_toc.read_text(encoding="utf-8", errors="ignore")
        require("'_pytest'" not in pyz_text and ".conftest'" not in pyz_text,
                "unexpected pytest/conftest modules bundled")
    return f"exe={exe.stat().st_size}"


def setup_and_requirements_contract():
    setup = (ROOT / "setup.bat").read_text(encoding="utf-8", errors="ignore")
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8", errors="ignore")
    build_req = (ROOT / "requirements-build.txt").read_text(encoding="utf-8", errors="ignore")
    tools_req = (ROOT / "requirements-tools.txt").read_text(encoding="utf-8", errors="ignore")
    req_pkgs = [
        line.strip().lower()
        for line in req.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    build_pkgs = [
        line.strip().lower()
        for line in build_req.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    tools_pkgs = [
        line.strip().lower()
        for line in tools_req.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    require("pip install" not in setup.replace("python -m pip install", ""),
            "setup.bat must use python -m pip install consistently")
    require("pyinstaller" not in req_pkgs, "pyinstaller should stay build-only")
    require("datasets" not in req_pkgs, "datasets should stay tools-only")
    require("ollama" not in req_pkgs, "ollama python package is not a runtime dependency")
    require("pyinstaller" in build_pkgs, "requirements-build.txt must include pyinstaller")
    require("datasets" in tools_pkgs, "requirements-tools.txt must include datasets")
    return "setup pip and dependency split contracts"


def first_run_readiness_contract():
    import shutil

    from koemo import readiness

    hub = TMP / "readiness_hub"
    models = TMP / "readiness_models"
    shutil.rmtree(hub, ignore_errors=True)
    shutil.rmtree(models, ignore_errors=True)
    hub.mkdir(parents=True, exist_ok=True)
    cfg = {
        "whisper_model": "large-v3-turbo",
        "summary_backend": "local",
        "fast_summary": True,
        "enable_diarization": True,
    }
    statuses = readiness.model_statuses(cfg, hub_dir=hub, models_dir=models)
    require(any(s["required"] and not s["ok"] for s in statuses), "missing Whisper should be surfaced")
    require("setup.bat" in readiness.first_run_notice(cfg, hub_dir=hub, models_dir=models),
            "first-run notice should include setup.bat recovery")
    msg = readiness.model_load_error("文字起こし", "large-v3-turbo", "network failed")
    require("setup.bat" in msg and "network failed" in msg, "model load recovery message incomplete")

    snap = hub / "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "model.bin").write_bytes(b"x")
    (models / "seg" / "model.int8.onnx").parent.mkdir(parents=True)
    (models / "seg" / "model.int8.onnx").write_bytes(b"x")
    (models / "emb").mkdir(parents=True)
    (models / "emb" / "speakernet.onnx").write_bytes(b"x")
    statuses = readiness.model_statuses(cfg, hub_dir=hub, models_dir=models)
    require(any(s["name"].startswith("Whisper") and s["ok"] for s in statuses), "cached Whisper not detected")
    require(any(s["name"] == "話者分離モデル" and s["ok"] for s in statuses), "diarization cache not detected")
    return "missing model notice + cached model detection"


def release_packaging_contract():
    script = ROOT / "scripts" / "koemo_release_build.py"
    iss = ROOT / "packaging" / "koemo.iss"
    notes = ROOT / "packaging" / "release-notes-ja.md"
    require(script.is_file(), "release build script missing")
    require(iss.is_file(), "Inno Setup script missing")
    require(notes.is_file(), "Japanese release notes missing")
    script_text = script.read_text(encoding="utf-8", errors="ignore")
    iss_text = iss.read_text(encoding="utf-8", errors="ignore")
    notes_text = notes.read_text(encoding="utf-8", errors="ignore")
    require("KOEMO_SIGNING_METADATA" in script_text, "Azure signing metadata env gate missing")
    require("KOEMO_SIGNING_DLIB" in script_text and "signtool" in script_text.lower(), "signing tool gate missing")
    require("--unsigned-beta" in script_text and "UNSIGNED-BETA" in script_text, "unsigned beta must be explicit")
    require("collect_pe_files" in script_text and '".pyd"' in script_text, "PE signing coverage missing")
    require("SHA256SUMS.txt" in script_text and "secret_scan" in script_text, "release checksum/secret gates missing")
    require("JRSoftware.InnoSetup" in script_text, "official Inno winget id missing")
    require("PrivilegesRequired=lowest" in iss_text, "installer should default to per-user install")
    require("compiler:Languages\\Japanese.isl" in iss_text, "Japanese installer language missing")
    require("..\\dist\\Koemo\\*" in iss_text, "installer must package PyInstaller dist")
    require("{{SIGNED_STATUS}}" in notes_text and "プライバシー" in notes_text, "release notes privacy/signing placeholders missing")
    return "signed release script + installer + Japanese notes contract"


def native_confidence_gate():
    from koemo.app import KoemoApp
    from koemo.config import MIC_LABEL
    from koemo.native_correction import normalize_transcript_text

    obj = KoemoApp.__new__(KoemoApp)
    bad = [(0.0, MIC_LABEL, "フッ素年五月")]
    good = [(0.0, MIC_LABEL, "これはテストです")]
    require(obj._native_rows_quality(bad, avg_conf=0.058) < 4.0, "low confidence garbage accepted")
    require(obj._native_rows_quality(good, avg_conf=0.6) >= 4.0, "reasonable native text rejected")
    require(normalize_transcript_text("これは声も高速化テストです") == "これはコエモ高速化テストです",
            "Koemo final correction missing")
    require(normalize_transcript_text("これはこれも高速化テストです") == "これはコエモ高速化テストです",
            "Koemo final correction for koremo missing")
    return "low-confidence System.Speech fragments are rejected"


def fast_integration_process_guards():
    import types

    import scripts.koemo_fast_integration_check as fast

    root = fast.ROOT.resolve()
    cases = [
        ({"name": "Koemo.exe", "exe": str(root / "dist" / "Koemo" / "Koemo.exe"), "cmdline": []}, True),
        ({"name": "pythonw.exe", "exe": r"C:\Python314\pythonw.exe", "cmdline": [str(root / "koemo.pyw")]}, True),
        ({"name": "python.exe", "exe": r"C:\Python314\python.exe", "cmdline": ["koemo.pyw"], "cwd": str(root)}, True),
        ({"name": "python.exe", "exe": r"C:\Python314\python.exe", "cmdline": ["koemo.pyw"], "cwd": str(root.parent / "other")}, False),
        ({"name": "pythonw.exe", "exe": r"C:\Python314\pythonw.exe", "cmdline": [str(root / "koemo.pyws")]}, False),
        ({"name": "pythonw.exe", "exe": r"C:\Python314\pythonw.exe", "cmdline": [str(root.parent / "other" / "koemo.pyw")]}, False),
        ({"name": "notepad.exe", "exe": str(root / "dist" / "Koemo" / "Koemo.exe"), "cmdline": []}, False),
        ({"name": "pythonw.exe", "exe": r"C:\Python314\pythonw.exe", "cmdline": ["other.pyw"], "cwd": str(root)}, False),
        ({"name": "python.exe", "exe": str(root / ".venv" / "Scripts" / "python.exe"), "cmdline": ["unrelated.py"], "cwd": str(root)}, False),
    ]
    for info, expected in cases:
        got = fast._is_koemo_process(info, root)
        require(got is expected, f"_is_koemo_process({info}) -> {got}, expected {expected}")

    class TimeoutExpired(Exception):
        pass

    class NoSuchProcess(Exception):
        pass

    calls = []

    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid
            self.waits = 0
        def terminate(self):
            calls.append(("terminate", self.pid))
        def wait(self, timeout):
            calls.append(("wait", self.pid, timeout))
            self.waits += 1
            if self.pid == 2 and self.waits == 1:
                raise TimeoutExpired()
        def kill(self):
            calls.append(("kill", self.pid))

    original_psutil = fast.psutil
    try:
        fast.psutil = types.SimpleNamespace(
            Process=lambda pid: (_ for _ in ()).throw(NoSuchProcess()) if pid == 404 else FakeProcess(pid),
            NoSuchProcess=NoSuchProcess,
            TimeoutExpired=TimeoutExpired,
        )
        fast._terminate(1)
        fast._terminate(2)
        fast._terminate(404)
    finally:
        fast.psutil = original_psutil

    require(("terminate", 1) in calls and ("wait", 1, 5) in calls, "terminate path did not wait")
    require(("terminate", 2) in calls and ("kill", 2) in calls and ("wait", 2, 5) in calls,
            "kill fallback path did not wait")

    cfg_path = fast.CONFIG
    original = cfg_path.read_text(encoding="utf-8") if cfg_path.is_file() else None
    try:
        require(fast.restore_config(True, '{"sentinel": true}') is True, "restore_config returned false")
        require(cfg_path.read_text(encoding="utf-8") == '{"sentinel": true}', "restore_config did not restore text")
        require(fast.restore_config(False, "") is True, "restore_config remove returned false")
        require(not cfg_path.exists(), "restore_config did not remove test-created config")
        cfg = fast.configure_fast_mode()
        for key, expected in {
            "record_mic": True,
            "record_system": True,
            "enable_aec": True,
            "final_channel_policy": "auto_dedupe",
        }.items():
            require(cfg.get(key) is expected or cfg.get(key) == expected,
                    f"configure_fast_mode did not force {key}: {cfg.get(key)!r}")
    finally:
        if original is not None:
            cfg_path.write_text(original, encoding="utf-8")
        else:
            try:
                cfg_path.unlink()
            except FileNotFoundError:
                pass

    report_paths = [fast.REPORT, fast.PYTHONW_REPORT, fast.EXE_REPORT]
    report_backups = {p: (p.read_text(encoding="utf-8") if p.is_file() else None) for p in report_paths}
    old_request_exe = fast.REQUEST_EXE
    old_require_exe = fast.REQUIRE_EXE
    try:
        report = {}
        fast.REQUEST_EXE = False
        fast.REQUIRE_EXE = False
        fast.write_report(report)
        require("config_restored_by_script" not in report, "write_report claimed config restoration before restore")
        original_exe = fast.EXE
        fast.EXE = fast.ROOT / "dist" / "Koemo" / "missing-koemo.exe"
        fast.validate_requested_exe()
        fast.REQUEST_EXE = True
        missing_failed = False
        try:
            fast.validate_requested_exe()
        except SystemExit:
            missing_failed = True
        require(missing_failed, "missing EXE should fail only when EXE launch is requested")
        fast.REQUEST_EXE = False
        fast.REQUIRE_EXE = True
        missing_required_failed = False
        try:
            fast.validate_requested_exe()
        except SystemExit:
            missing_required_failed = True
        require(missing_required_failed, "missing EXE should fail when EXE is required")
    finally:
        fast.REQUEST_EXE = old_request_exe
        fast.REQUIRE_EXE = old_require_exe
        if "original_exe" in locals():
            fast.EXE = original_exe
        for path, text in report_backups.items():
            if text is None:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            else:
                path.write_text(text, encoding="utf-8")

    spool_dir = fast.SAVE_DIR / ".koemo_tmp"
    spool_dir.mkdir(parents=True, exist_ok=True)
    stale_spool = spool_dir / "capture_smoke_stale.f32"
    stale_spool.write_bytes(b"stale")
    removed = fast.cleanup_test_spool_files({"save_dir": str(fast.SAVE_DIR)})
    require(str(stale_spool) in removed, f"integration stale spool was not reported removed: {removed}")
    require(not stale_spool.exists(), "integration stale spool was not removed")

    good_summary = (
        "# これはコエモ高速化テストです\n\n"
        "## 要旨\nこれはコエモ高速化テストです。停止後10秒以内の処理です。\n\n"
        "---\n\n## 文字起こし\n\n"
        "**相手**: これはコエモ高速化テストです。\n"
        "**相手**: ライブ文字起こしと停止後10秒以内の処理を確認しています。\n"
    )
    good_eval = fast.evaluate_summary(good_summary)
    require(good_eval["has_whisper_final"], f"valid transcript section rejected: {good_eval}")
    summary_only = (
        "# これはコエモ高速化テストです\n\n"
        "## 要旨\nこれはコエモ高速化テストです。停止後10秒以内の処理です。\n\n"
        "---\n\n## 文字起こし\n\n"
        "**相手**: 別の短い発話です。\n"
    )
    bad_eval = fast.evaluate_summary(summary_only)
    require(not bad_eval["has_expected_text"] and not bad_eval["has_whisper_final"],
            f"summary-body text leaked into transcript verdict: {bad_eval}")
    placeholder = "## 文字起こし\n\n**あなた**: Windows音声認識の確定結果がありません。\n"
    placeholder_eval = fast.evaluate_summary(placeholder)
    require(not placeholder_eval["has_whisper_final"], f"native placeholder accepted: {placeholder_eval}")
    hallucination = (
        "## 文字起こし\n\n"
        "**あなた**: ご視聴ありがとうございました\n"
        "**相手**: これはコエモ高速化テストです。停止後10秒以内の処理です。\n"
    )
    hallucination_eval = fast.evaluate_summary(hallucination)
    require(hallucination_eval["has_common_hallucination"] and not hallucination_eval["has_whisper_final"],
            f"common hallucination accepted in integration verdict: {hallucination_eval}")

    return "process matcher 9 cases; terminate waits; config restore works; report restore claim deferred; test spool cleanup and transcript section verdict work"


def _latest_wav_real_transcribe_summary_impl():
    from koemo.config import DEFAULT_CONFIG, MIC_LABEL
    from koemo.live import NativeWindowsLiveTranscriber
    from koemo.summarize import Summarizer
    from koemo.transcribe import merge_rows

    require(NativeWindowsLiveTranscriber.available(), "Windows native speech recognizer unavailable")
    transcript = merge_rows([(0.0, MIC_LABEL, "これはコエモの実機検証です。Windows純正音声認識で文字起こしします。")])
    cfg = DEFAULT_CONFIG.copy()
    cfg["summary_backend"] = "local"
    s = Summarizer("", 300, False, cfg)
    title, body = s.fast_summarize(transcript, language="ja")
    require(title and body, "summary empty")
    out = TMP / "real_recording_summary.md"
    out.write_text(f"# {title}\n\n{body}\n\n---\n\n## 文字起こし\n\n{transcript}\n", encoding="utf-8")
    return f"hybrid_live_native=True transcript_chars={len(transcript)} output={out}"


def latest_wav_real_transcribe_summary():
    env = os.environ.copy()
    env["KOEMO_SMOKE_ISOLATED_LATEST"] = "1"
    proc = subprocess.run(
        [sys.executable, __file__, "--latest-only"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
    )
    require(proc.returncode == 0, (proc.stderr or proc.stdout).strip())
    return proc.stdout.strip()


def launch_pythonw():
    proc = subprocess.Popen(["pythonw", "koemo.pyw"], cwd=ROOT)
    time.sleep(7)
    alive = proc.poll() is None
    if alive:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    require(alive, "pythonw app exited")
    return f"pid={proc.pid}"


def launch_exe():
    exe = ROOT / "dist" / "Koemo" / "Koemo.exe"
    try:
        proc = subprocess.Popen([str(exe)], cwd=exe.parent)
        mode = "exe"
    except OSError as e:
        require(getattr(e, "winerror", None) == 4551, f"exe launch failed: {e}")
        proc = subprocess.Popen(["pythonw", "koemo.pyw"], cwd=ROOT)
        mode = "pythonw fallback after Windows App Control blocked unsigned exe"
    time.sleep(7)
    alive = proc.poll() is None
    if alive:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    require(alive, "exe app exited")
    return f"mode={mode} pid={proc.pid}"


def main():
    tests = [
        ("dependencies", deps),
        ("config_defaults", config_defaults),
        ("no_forbidden_qt_exec_pattern", no_exec_pattern),
        ("audio_snapshot_live", audio_snapshot_live),
        ("live_fallback_contract", live_fallback_contract),
        ("live_backend_auto_resolves_by_gpu_contract", live_backend_auto_resolves_by_gpu_contract),
        ("live_start_nonblocking_contract", live_start_nonblocking_contract),
        ("live_async_failure_falls_back_contract", live_async_failure_falls_back_contract),
        ("native_state_detected_event_contract", native_state_detected_event_contract),
        ("live_activity_prearms_before_native_contract", live_activity_prearms_before_native_contract),
        ("native_live_respects_record_mic_contract", native_live_respects_record_mic_contract),
        ("final_ignores_live_rows_contract", final_ignores_live_rows_contract),
        ("final_drops_saturated_mic_echo_contract", final_drops_saturated_mic_echo_contract),
        ("final_audio_finite_samples_contract", final_audio_finite_samples_contract),
        ("final_filters_common_whisper_hallucination_contract", final_filters_common_whisper_hallucination_contract),
        ("aec_batched_wiener_contract", aec_batched_wiener_contract),
        ("recording_capture_errors_surface_contract", recording_capture_errors_surface_contract),
        ("recording_save_dir_failure_nonfatal_contract", recording_save_dir_failure_nonfatal_contract),
        ("recording_all_capture_errors_surface_contract", recording_all_capture_errors_surface_contract),
        ("recording_no_channel_config_rejected_contract", recording_no_channel_config_rejected_contract),
        ("recorder_spools_audio_and_bounds_live_contract", recorder_spools_audio_and_bounds_live_contract),
        ("recorder_hung_thread_reports_temp_contract", recorder_hung_thread_reports_temp_contract),
        ("settings_save_during_recording_contract", settings_save_during_recording_contract),
        ("full_summary_releases_whisper_models_contract", full_summary_releases_whisper_models_contract),
        ("import_full_summary_releases_whisper_models_contract", import_full_summary_releases_whisper_models_contract),
        ("import_save_dir_failure_falls_back_contract", import_save_dir_failure_falls_back_contract),
        ("process_cleans_spool_files_contract", process_cleans_spool_files_contract),
        ("process_surfaces_spool_cleanup_failure_contract", process_surfaces_spool_cleanup_failure_contract),
        ("preload_state_contract", preload_state_contract),
        ("real_short_recording", real_short_recording),
        ("summarizer_templates_chat_backends", summarizer_templates_chat_backends),
        ("library_roundtrip", library_roundtrip),
        ("calendar_title_hint_contract", calendar_title_hint_contract),
        ("diarize_models", diarize_models),
        ("exports_and_ui", exports_and_ui),
        ("meeting_detection", meeting_detection),
        ("packaging_files", packaging_files),
        ("setup_and_requirements_contract", setup_and_requirements_contract),
        ("first_run_readiness_contract", first_run_readiness_contract),
        ("release_packaging_contract", release_packaging_contract),
        ("native_confidence_gate", native_confidence_gate),
        ("fast_integration_process_guards", fast_integration_process_guards),
        ("latest_wav_real_transcribe_summary", latest_wav_real_transcribe_summary),
        ("launch_pythonw", launch_pythonw),
        ("launch_exe", launch_exe),
    ]
    for name, func in tests:
        check(name, func)

    report_json = TMP / "report.json"
    report_md = TMP / "report.md"
    report_json.write_text(json.dumps(RESULTS, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Koemo feature smoke report", ""]
    for r in RESULTS:
        mark = "x" if r["status"] == "pass" else "!"
        lines.append(f"- [{mark}] {r['name']} ({r['seconds']}s): {r['detail']}")
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report_json={report_json}")
    print(f"report_md={report_md}")
    fails = [r for r in RESULTS if r["status"] != "pass"]
    return 1 if fails else 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--latest-only":
        print(_latest_wav_real_transcribe_summary_impl())
        raise SystemExit(0)
    raise SystemExit(main())
