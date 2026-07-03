"""Koemo EXE を実起動し、停止後10秒以内に結果が出るかを測る。"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import keyboard
import psutil


ROOT = Path(__file__).resolve().parents[1]
EXE = Path(os.environ.get("KOEMO_TEST_EXE_PATH", str(ROOT / "dist" / "Koemo" / "Koemo.exe")))
CONFIG = Path.home() / ".koemo" / "config.json"
SAVE_DIR = ROOT / "outputs" / "meetings"
FALLBACK_DIR = Path.home() / "AppData" / "Local" / "Koemo" / "Recordings"
REPORT = ROOT / ".codex_tmp" / "feature_smoke" / "fast_integration_report.json"
PYTHONW_REPORT = ROOT / ".codex_tmp" / "feature_smoke" / "fast_integration_pythonw_report.json"
EXE_REPORT = ROOT / ".codex_tmp" / "feature_smoke" / "fast_integration_exe_report.json"
COMMAND_FILE = ROOT / ".codex_tmp" / "feature_smoke" / "fast_integration_command.txt"
SEND_HOTKEY = str(os.environ.get("KOEMO_TEST_SEND_HOTKEY", "0")).lower() in {"1", "true", "yes"}
REQUEST_EXE = str(os.environ.get("KOEMO_TEST_EXE", "0")).lower() in {"1", "true", "yes"}
REQUIRE_EXE = str(os.environ.get("KOEMO_TEST_REQUIRE_EXE", "0")).lower() in {"1", "true", "yes"}
MAX_STOP_SECONDS = float(os.environ.get("KOEMO_TEST_MAX_SECONDS", "10"))


def _is_koemo_process(info, root_resolved):
    """プロセスが Koemo 本体（ROOT配下の Koemo.exe か、pythonw/python の
    koemo.pyw 起動）かをパス境界で判定する。部分一致(koemo.pyws 等)や別作業
    ツリーのパスを取り込まないよう、basename 一致と is_relative_to で見る。"""
    name = (info.get("name") or "").lower()
    if name not in {"koemo.exe", "pythonw.exe", "python.exe"}:
        return False
    exe = info.get("exe") or ""
    if name == "koemo.exe" and exe:
        try:
            if Path(exe).resolve().is_relative_to(root_resolved):
                return True
        except (OSError, ValueError):
            pass
    for arg in (info.get("cmdline") or []):
        try:
            p = Path(arg)
            if p.name.lower() != "koemo.pyw":
                continue
            if p.is_absolute():
                candidate = p.resolve()
            else:
                cwd = info.get("cwd")
                if not cwd:
                    continue
                candidate = (Path(cwd) / p).resolve()
            if candidate.is_relative_to(root_resolved):
                return True
        except (OSError, ValueError):
            continue
    return False


def _terminate(pid):
    """pid を terminate→wait、ダメなら kill→wait で確実に終了させる。"""
    try:
        p = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    try:
        p.terminate()
        p.wait(timeout=5)
    except psutil.TimeoutExpired:
        try:
            p.kill()
            p.wait(timeout=5)
        except Exception:
            pass
    except Exception:
        pass


def stop_existing():
    root_resolved = ROOT.resolve()
    self_pid = os.getpid()
    stopped = []
    for proc in psutil.process_iter(["pid", "name", "exe", "cmdline", "cwd"]):
        try:
            if proc.info["pid"] == self_pid:
                continue
            if not _is_koemo_process(proc.info, root_resolved):
                continue
            stopped.append({"pid": proc.info["pid"], "name": proc.info["name"], "exe": proc.info.get("exe")})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    for item in stopped:
        _terminate(item["pid"])
    return stopped


def launch_koemo():
    env = os.environ.copy()
    env["KOEMO_TEST_COMMAND_FILE"] = str(COMMAND_FILE)
    # 10秒目標は Windows Security 環境で推奨する pythonw/start.bat 経路で測る。
    # PyInstaller EXE はCUDA DLLを同梱しない便利配布用なので、Whisper final がCPUに落ちることがある。
    if not REQUEST_EXE:
        return subprocess.Popen(["pythonw", "koemo.pyw"], cwd=str(ROOT), env=env), "pythonw"
    try:
        return subprocess.Popen([str(EXE)], cwd=str(EXE.parent), env=env), "exe"
    except OSError as e:
        if getattr(e, "winerror", None) != 4551:
            raise
        if REQUIRE_EXE:
            raise RuntimeError(f"EXE blocked by Windows App Control: {e}")
        return subprocess.Popen(["pythonw", "koemo.pyw"], cwd=str(ROOT), env=env), "pythonw"


def restore_config(existed, text):
    if existed:
        CONFIG.write_text(text, encoding="utf-8")
    else:
        try:
            CONFIG.unlink()
        except FileNotFoundError:
            pass
    if existed:
        return CONFIG.is_file() and CONFIG.read_text(encoding="utf-8") == text
    return not CONFIG.exists()


def write_report(report):
    report["requested_exe"] = REQUEST_EXE
    report["require_exe"] = REQUIRE_EXE
    report["exe_path"] = str(EXE)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if REQUEST_EXE or REQUIRE_EXE or report.get("launch_mode") == "exe":
        EXE_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        PYTHONW_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_markdown_section(text, heading):
    match = re.search(rf"(?m)^##\s+{re.escape(heading)}\s*$", text)
    if not match:
        return ""
    start = match.end()
    next_heading = re.search(r"(?m)^##\s+", text[start:])
    end = start + next_heading.start() if next_heading else len(text)
    return text[start:end].strip()


def evaluate_summary(text):
    transcript = extract_markdown_section(text, "文字起こし")
    has_transcript_section = len(transcript) > 20
    has_transcript_label = bool(re.search(r"^\*\*[^*]+\*\*:", transcript, re.MULTILINE))
    has_expected_text = "コエモ" in transcript and ("停止後" in transcript or "10秒" in transcript or "十秒" in transcript)
    has_native_placeholder = "Windows音声認識の確定結果がありません" in transcript
    has_common_hallucination = "ご視聴ありがとうございました" in transcript or "ご清聴ありがとうございました" in transcript
    # This deterministic check plays TTS through the system output only. A mic row
    # here means the final path accepted speaker leak/echo as local speech.
    has_unexpected_mic_row = bool(re.search(r"^\*\*あなた\*\*:", transcript, re.MULTILINE))
    has_whisper_final = (has_transcript_section and has_transcript_label and has_expected_text
                         and not has_native_placeholder and not has_common_hallucination
                         and not has_unexpected_mic_row)
    return {
        "transcript": transcript,
        "has_transcript_section": has_transcript_section,
        "has_transcript_label": has_transcript_label,
        "has_expected_text": has_expected_text,
        "has_common_hallucination": has_common_hallucination,
        "has_unexpected_mic_row": has_unexpected_mic_row,
        "has_whisper_final": has_whisper_final,
    }


def configure_fast_mode():
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    cfg = {}
    if CONFIG.is_file():
        try:
            cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
    cfg.update({
        "enable_live_transcription": True,
        "native_only_transcription": False,
        "live_backend": "native_windows",
        "live_fallback_backend": "whisper_rolling",
        "native_speech_language": "ja-JP",
        "whisper_model": "large-v3-turbo",
        "live_whisper_model": "small",
        "fast_summary": True,
        "use_live_transcript_on_stop": False,
        "preload_transcriber": True,
        "preload_final_transcriber": True,
        "record_mic": True,
        "record_system": True,
        "enable_aec": True,
        "final_channel_policy": "auto_dedupe",
        "save_dir": str(SAVE_DIR),
        "live_interval_sec": 1.2,
        "live_window_sec": 8.0,
        "live_stable_margin_sec": 1.5,
        "live_min_audio_sec": 0.8,
    })
    CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return cfg


def speak(text):
    escaped = text.replace("'", "''")
    script = f"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {{ $s.SelectVoice('Microsoft Haruka Desktop') }} catch {{}}
$s.Rate = -1
$s.Volume = 100
$s.Speak('{escaped}')
$s.Dispose()
""".strip()
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        check=True,
        timeout=45,
    )


def send_toggle(seq, hotkey):
    COMMAND_FILE.parent.mkdir(parents=True, exist_ok=True)
    COMMAND_FILE.write_text(f"toggle:{seq}:{time.time()}", encoding="utf-8")
    if not SEND_HOTKEY:
        return
    # 通常の統合チェックは deterministic な file command のみで操作する。
    # 実機ホットキー経路まで同時確認したい時だけ KOEMO_TEST_SEND_HOTKEY=1 で併用する。
    try:
        keyboard.press_and_release(hotkey)
    except Exception:
        pass


def control_mode():
    return "test_command_file+hotkey" if SEND_HOTKEY else "test_command_file"


def summary_dirs(cfg):
    dirs = [Path(cfg.get("save_dir") or SAVE_DIR), SAVE_DIR, FALLBACK_DIR]
    out = []
    for directory in dirs:
        if directory not in out:
            out.append(directory)
    return out


def cleanup_test_spool_files(cfg):
    """Integration testが強制終了したKoemoの録音spoolだけを掃除する。"""
    removed = []
    for directory in summary_dirs(cfg):
        tmp_dir = directory / ".koemo_tmp"
        if not tmp_dir.is_dir():
            continue
        for path in tmp_dir.glob("capture_*.f32"):
            if not path.is_file():
                continue
            try:
                path.unlink()
                removed.append(str(path))
            except Exception:
                pass
    return removed


def wait_for_new_summary(before, stop_time, cfg, timeout=20):
    for directory in summary_dirs(cfg):
        directory.mkdir(parents=True, exist_ok=True)
    FALLBACK_DIR.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        current = []
        for directory in summary_dirs(cfg):
            current.extend(directory.glob("summary_*.md"))
        current = sorted(current, key=lambda p: p.stat().st_mtime, reverse=True)
        for path in current:
            if str(path) in before:
                continue
            if path.stat().st_mtime >= stop_time - 1:
                return path
        time.sleep(0.25)
    return None


def validate_requested_exe():
    if (REQUEST_EXE or REQUIRE_EXE) and not EXE.is_file():
        raise SystemExit(f"EXE missing: {EXE}")


def main():
    validate_requested_exe()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    original_config_exists = CONFIG.is_file()
    original_config_text = CONFIG.read_text(encoding="utf-8") if original_config_exists else ""
    try:
        COMMAND_FILE.unlink()
    except FileNotFoundError:
        pass

    proc = None
    cfg = None
    last_report = None
    try:
        stopped = stop_existing()
        cfg = configure_fast_mode()
        cleanup_test_spool_files(cfg)
        hotkey = cfg.get("hotkey", "ctrl+shift+r")
        before = set()
        for directory in summary_dirs(cfg):
            if directory.exists():
                before.update(str(p) for p in directory.glob("summary_*.md"))

        proc, launch_mode = launch_koemo()
        time.sleep(60)

        if proc.poll() is not None:
            raise SystemExit(f"Koemo exited before recording start: exit={proc.returncode}")
        send_toggle("start", hotkey)
        time.sleep(1.5)
        speak("これはコエモ高速化テストです。ライブ文字起こしと停止後十秒以内の処理を確認しています。")
        time.sleep(1.0)

        stop_time = time.time()
        send_toggle("stop", hotkey)
        wait_timeout = max(20.0, MAX_STOP_SECONDS + 5.0)
        summary = wait_for_new_summary(before, stop_time, cfg, timeout=wait_timeout)
        elapsed = time.time() - stop_time
        if not summary:
            report = {
                "ok": False,
                "stop_to_summary_seconds": round(elapsed, 3),
                "summary": None,
                "note": f"summary file was not created within {wait_timeout:.0f} seconds",
                "app_pid": proc.pid,
                "launch_mode": launch_mode,
                "control_mode": control_mode(),
                "stopped_processes": stopped,
                "hotkey": hotkey,
                "command_file": str(COMMAND_FILE),
                "max_stop_seconds": MAX_STOP_SECONDS,
            }
            last_report = report
            write_report(report)
            raise SystemExit(f"summary file was not created within {wait_timeout:.0f} seconds")

        text = summary.read_text(encoding="utf-8", errors="ignore")
        evaluation = evaluate_summary(text)
        transcript = evaluation["transcript"]
        has_transcript_section = evaluation["has_transcript_section"]
        has_whisper_final = evaluation["has_whisper_final"]
        has_expected_text = evaluation["has_expected_text"]
        has_transcript_label = evaluation["has_transcript_label"]
        has_common_hallucination = evaluation["has_common_hallucination"]
        has_unexpected_mic_row = evaluation["has_unexpected_mic_row"]
        ok = elapsed <= MAX_STOP_SECONDS and has_whisper_final
        report = {
            "ok": ok,
            "stop_to_summary_seconds": round(elapsed, 3),
            "summary": str(summary),
            "summary_chars": len(text),
            "transcript_chars": len(transcript),
            "has_whisper_final": has_whisper_final,
            "has_expected_text": has_expected_text,
            "has_transcript_section": has_transcript_section,
            "has_transcript_label": has_transcript_label,
            "has_common_hallucination": has_common_hallucination,
            "has_unexpected_mic_row": has_unexpected_mic_row,
            "note": "" if has_whisper_final else "final transcript section did not contain the expected labelled Whisper text",
            "app_pid": proc.pid,
            "launch_mode": launch_mode,
            "control_mode": control_mode(),
            "stopped_processes": stopped,
            "hotkey": hotkey,
            "command_file": str(COMMAND_FILE),
            "max_stop_seconds": MAX_STOP_SECONDS,
            "first_lines": text.splitlines()[:12],
            "transcript_first_lines": transcript.splitlines()[:8],
        }
        last_report = report
        write_report(report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if ok else 1
    finally:
        # このスクリプトが起動した Koemo を残存させない（次回実行のホットキー競合防止）。
        if proc is not None:
            _terminate(proc.pid)
        removed_spool = cleanup_test_spool_files(cfg) if cfg is not None else []
        restored = restore_config(original_config_exists, original_config_text)
        if last_report is not None:
            last_report["test_spool_removed_by_script"] = removed_spool
            last_report["config_restored_by_script"] = restored
            write_report(last_report)


if __name__ == "__main__":
    raise SystemExit(main())
