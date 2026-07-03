"""Koemo 設定の読み書きと既定値。"""
import json
import os
from datetime import datetime
from pathlib import Path

APP_NAME       = "Koemo"
CONFIG_DIR     = Path.home() / ".koemo"
CONFIG_FILE    = CONFIG_DIR / "config.json"
RECORDINGS_DIR = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Koemo" / "Recordings"

# 話者ラベル（mic=自分 / system=相手）
MIC_LABEL = "あなた"
SYS_LABEL = "相手"

DEFAULT_CONFIG = {
    "summary_model_dir": "",        # 要約用CT2モデルのパス（空欄で自動検出）
    "summary_backend":   "local",   # local / ollama / openai_compat
    "ollama_model":      "qwen2.5:3b",
    "ollama_base_url":   "http://localhost:11434",
    "openai_base_url":   "",
    "openai_api_key":    "",
    "openai_model":      "",
    "summary_sections":  ["要旨", "主要トピック", "決定事項", "アクションアイテム", "未解決の質問"],
    "summary_extra_instructions": "",
    "whisper_model":     "large-v3-turbo",  # 最終処理用: tiny/base/small/medium/large-v3-turbo/large-v3
    "live_whisper_model": "small",          # ライブ用: 速度と日本語精度のバランス
    "hotkey":            "ctrl+shift+r",
    "save_dir":          str(RECORDINGS_DIR),
    "summary_language":  "ja",
    "sample_rate":       16000,
    "cpu_threads":       2,
    "idle_unload_sec":   300,
    "keep_warm":         False,     # 真でモデルをアイドル解放しない（連続会議向け・高速だがVRAM占有）
    "preload_transcriber": True,     # 起動後に文字起こしモデルを先読みして初回遅延を減らす
    "preload_final_transcriber": True,  # 最終用largeモデルも裏で先読みする
    "show_model_ready_status": True,  # 起動時/録音前に高精度モデルの準備状態を表示
    "fast_summary":      True,      # 停止後10秒以内を優先し、軽量な即時要約を使う
    "use_live_transcript_on_stop": False,  # 精度優先: 停止後は最終用モデルで取り直す
    "live_backend":      "auto",  # auto(GPUあり=whisper_rolling/無し=native_windows) / native_windows / whisper_rolling / off
    "live_fallback_backend": "whisper_rolling",
    "native_speech_language": "ja-JP",
    "native_speech_startup_settle_sec": 1.0,
    "final_channel_policy": "auto_dedupe",  # auto_dedupe / all_active
    "native_only_transcription": False,  # 実験用: TrueならWindows純正だけで正式 transcript を作る
    "live_interval_sec": 1.2,
    "live_window_sec":   8.0,
    "live_stable_margin_sec": 1.5,
    "live_min_audio_sec": 0.8,
    "record_mic":        True,
    "record_system":     True,
    "enable_aec":        True,
    "enable_diarization": True,     # システム音声側の話者分離（相手1/相手2…）
    "enable_live_transcription": True,  # 録音中のライブ文字起こしプレビュー
    "enable_meeting_detection": True,   # Zoom/Teams等の起動を通知
    "enable_calendar_title_hint": False,  # ICS/Outlook予定タイトルを会議名の既定値に使う
    "calendar_ics_path": "",
    "calendar_outlook_enabled": False,
    "calendar_title_lookback_min": 15,
    "calendar_title_lookahead_min": 10,
    "mic_name":          "",
    "speaker_name":      "",
}

_BOOL_KEYS = {
    "keep_warm", "preload_transcriber", "preload_final_transcriber",
    "show_model_ready_status", "fast_summary", "use_live_transcript_on_stop",
    "native_only_transcription", "record_mic", "record_system", "enable_aec",
    "enable_diarization", "enable_live_transcription", "enable_meeting_detection",
    "enable_calendar_title_hint", "calendar_outlook_enabled",
}

_INT_KEYS = {
    "sample_rate", "cpu_threads", "idle_unload_sec",
    "calendar_title_lookback_min", "calendar_title_lookahead_min",
}

_FLOAT_KEYS = {
    "native_speech_startup_settle_sec", "live_interval_sec", "live_window_sec",
    "live_stable_margin_sec", "live_min_audio_sec",
}

_NONNEGATIVE_KEYS = {
    "sample_rate", "cpu_threads", "idle_unload_sec", "calendar_title_lookback_min",
    "calendar_title_lookahead_min", "native_speech_startup_settle_sec",
    "live_interval_sec", "live_window_sec", "live_stable_margin_sec", "live_min_audio_sec",
}


def _coerce_config(raw):
    cfg = DEFAULT_CONFIG.copy()
    if not isinstance(raw, dict):
        raise ValueError("config root must be a JSON object")
    for key, value in raw.items():
        if key not in DEFAULT_CONFIG:
            continue
        default = DEFAULT_CONFIG[key]
        try:
            if key in _BOOL_KEYS:
                if isinstance(value, bool):
                    cfg[key] = value
                elif isinstance(value, str) and value.strip().lower() in {"true", "false"}:
                    cfg[key] = value.strip().lower() == "true"
                else:
                    cfg[key] = bool(value)
            elif key in _INT_KEYS:
                coerced = int(value)
                if key in _NONNEGATIVE_KEYS and coerced < 0:
                    coerced = default
                cfg[key] = coerced
            elif key in _FLOAT_KEYS:
                coerced = float(value)
                if key in _NONNEGATIVE_KEYS and coerced < 0:
                    coerced = default
                cfg[key] = coerced
            elif isinstance(default, list):
                if isinstance(value, list):
                    cfg[key] = [str(v) for v in value if str(v).strip()]
                elif isinstance(value, str):
                    cfg[key] = [s.strip() for s in value.split(",") if s.strip()]
            elif isinstance(default, str):
                cfg[key] = str(value)
            else:
                cfg[key] = value
        except Exception:
            cfg[key] = default
    return cfg


def load_config():
    CONFIG_DIR.mkdir(exist_ok=True)
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return _coerce_config(json.load(f))
        except Exception as e:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = CONFIG_FILE.with_name(f"config.invalid_{stamp}.json")
            try:
                CONFIG_FILE.replace(backup)
                backup_msg = str(backup)
            except Exception as move_error:
                backup_msg = f"backup failed: {move_error}"
            cfg = DEFAULT_CONFIG.copy()
            cfg["_config_load_error"] = f"{CONFIG_FILE}: {e}"
            cfg["_config_backup"] = backup_msg
            return cfg
    return DEFAULT_CONFIG.copy()


def save_config(cfg):
    CONFIG_DIR.mkdir(exist_ok=True)
    public_cfg = _coerce_config({k: v for k, v in dict(cfg).items() if not str(k).startswith("_")})
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(public_cfg, f, indent=2, ensure_ascii=False)
