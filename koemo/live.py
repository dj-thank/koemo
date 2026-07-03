"""録音中のライブ文字起こしプレビュー。

Kanary型の二段構成にするため、ライブ字幕は差し替え可能なバックエンドにする。
Windows純正 WinRT Speech で低遅延プレビューを作り、正式 transcript は停止後の高精度Whisperで作る。
古いSAPI/System.Speechは互換 fallback としてだけ使う。
"""
import json
import os
import asyncio
import platform
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np

from .config import MIC_LABEL, SYS_LABEL
from .native_correction import normalize_native_text


@dataclass(frozen=True)
class LiveEvent:
    start: float
    end: float
    label: str
    text: str
    kind: str = "final"  # final / provisional / status


def format_live_events(events):
    """LiveEvent列をライブウィンドウ向けのテキストにする。"""
    lines = []
    for ev in events:
        if not ev.text:
            continue
        if ev.kind == "status":
            lines.append(ev.text)
        elif ev.kind == "provisional":
            lines.append(f"{ev.label}: {ev.text} ...")
        else:
            lines.append(f"{ev.label}: {ev.text}")
    return "\n".join(lines)


class NativeWindowsLiveTranscriber:
    """Windows純正音声認識を使う低遅延マイク字幕。

    日本語自由発話は古いSAPI/System.Speechより WinRT Speech の方がましなため、
    WinRTを第一候補にし、失敗時だけ System.Speech ブリッジへ落とす。
    """

    def __init__(self, language="ja-JP", label=MIC_LABEL, on_update=None, on_error=None,
                 on_unavailable=None, startup_settle_sec=1.0):
        self.language = language or "ja-JP"
        self.label = label
        self.on_update = on_update
        self.on_error = on_error
        self.on_unavailable = on_unavailable
        self.startup_settle_sec = max(0.0, float(startup_settle_sec or 0.0))
        self._stop = threading.Event()
        self._thread = None
        self._rows_lock = threading.Lock()
        self._final = []
        self._hypothesis = ""
        self._speech_detected = False
        self._started = time.time()
        self._ready = threading.Event()
        self._startup_error = None
        self._proc = None

    @staticmethod
    def available():
        if platform.system() != "Windows":
            return False
        try:
            import winrt.windows.media.speechrecognition  # noqa: F401
            import winrt.windows.globalization  # noqa: F401
            return True
        except Exception:
            try:
                cmd = (
                    "Add-Type -AssemblyName System.Speech; "
                    "[System.Speech.Recognition.SpeechRecognitionEngine]::InstalledRecognizers() "
                    "| ForEach-Object { $_.Culture.Name }"
                )
                proc = subprocess.run(
                    ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", cmd],
                    text=True,
                    capture_output=True,
                    timeout=8,
                )
                return proc.returncode == 0 and bool(proc.stdout.strip())
            except Exception:
                return False

    def start(self, wait=False):
        if self._thread and self._thread.is_alive():
            return
        if platform.system() != "Windows":
            raise RuntimeError("Windows Speech は Windows でのみ利用できます")
        self._stop.clear()
        self._startup_error = None
        self._ready.clear()
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()
        # アプリ既定(wait=False)はGUIスレッドをブロックしない。起動失敗は
        # on_error / on_unavailable コールバック経由で fallback させる。
        # 計測ハーネスだけが wait=True で「認識開始」を待ってからTTSを流す。
        if wait:
            self._ready.wait(timeout=4.0)
            if self._startup_error:
                self.stop(timeout=0.2)
                raise RuntimeError(self._startup_error)

    def stop(self, timeout=1.0):
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.kill()
            except Exception:
                pass
        self._thread = None

    def final_rows(self):
        return [(ev.start, ev.label, ev.text) for ev in self.events() if ev.kind == "final"]

    def events(self):
        with self._rows_lock:
            rows = list(self._final)
            hypothesis = self._hypothesis.strip()
        out = list(rows)
        if hypothesis and (not out or out[-1].text != hypothesis):
            now = time.time() - self._started
            out.append(LiveEvent(max(0.0, now - 1.0), now, self.label, hypothesis, "provisional"))
        return out

    def _thread_main(self):
        try:
            try:
                asyncio.run(self._run_winrt_async())
            except Exception as winrt_error:
                if self._stop.is_set():
                    return
                try:
                    self._run_bridge()
                except Exception as bridge_error:
                    raise RuntimeError(
                        f"WinRT Speech failed: {winrt_error}; System.Speech failed: {bridge_error}"
                    ) from bridge_error
        except Exception as e:
            was_ready = self._ready.is_set()
            if not was_ready:
                self._startup_error = str(e)
            self._ready.set()
            if self._stop.is_set():
                return
            if self.on_error:
                self.on_error(str(e))
            # 起動前・起動後どちらの失敗でも fallback を促す。非ブロック起動では
            # start() が例外を投げないため、ready前失敗もここで通知する。
            if self.on_unavailable:
                self.on_unavailable(str(e))

    async def _run_winrt_async(self):
        try:
            from winrt.windows.globalization import Language
            from winrt.windows.media.speechrecognition import SpeechRecognizer
        except Exception as e:
            raise RuntimeError("PyWinRT SpeechRecognition package is not installed") from e

        recognizer = SpeechRecognizer(self._resolve_language(Language, SpeechRecognizer))
        recognizer.timeouts.initial_silence_timeout = timedelta(seconds=5)
        recognizer.timeouts.end_silence_timeout = timedelta(milliseconds=250)
        recognizer.timeouts.babble_timeout = timedelta(seconds=0)
        # Constraints improve command-style recognition but add latency in
        # continuous Japanese captions. Keep WinRT live unconstrained and apply
        # Koemo-specific vocabulary correction after each hypothesis/result.
        session = recognizer.continuous_recognition_session
        recognizer.add_hypothesis_generated(lambda _sender, args: self._on_hypothesis_text(
            getattr(getattr(args, "hypothesis", None), "text", "") or ""))
        # state_changed はエンジン由来の「発話検知」を返す。最初の文字仮説より
        # 早く届くので、ライブUIを動かす低遅延シグナルとして使う。
        recognizer.add_state_changed(lambda _sender, args: self._on_state(getattr(args, "state", None)))
        session.add_result_generated(lambda _sender, args: self._on_result_text(
            getattr(getattr(args, "result", None), "text", "") or ""))

        compile_result = await recognizer.compile_constraints_async()
        status = getattr(compile_result, "status", None)
        status_name = str(status).lower()
        if status is not None and "success" not in status_name and status_name not in {"0", "speechrecognitionresultstatus.success"}:
            raise RuntimeError(f"compile_constraints_async failed: {status}")

        await session.start_async()
        # エンジンは start_async 完了直後からキャプチャしている。「ライブ文字起こし中」は
        # 発話より前に一度だけ出し、以降は再送しない（発話検知/仮説を上書きしない）。
        if not (self._speech_detected or self._hypothesis or self._final):
            self._emit_status("ライブ文字起こし中")
        # settle は表示の安定化用。GUIは非ブロック起動なので影響しない。wait=True の
        # 計測ハーネスだけが _ready を待つため、ここで settle 後に _ready を立てる。
        if self.startup_settle_sec:
            await asyncio.sleep(self.startup_settle_sec)
        self._ready.set()
        while not self._stop.is_set():
            await asyncio.sleep(0.1)
        try:
            await session.cancel_async()
        except Exception:
            pass
        try:
            recognizer.close()
        except Exception:
            pass

    def _run_bridge(self):
        self._proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-File", self._bridge_path(), "-Language", self.language],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = event.get("type", "")
            if kind == "status" and event.get("text") == "ready":
                self._emit_status("ライブ文字起こし中")
                self._ready.set()
            elif kind == "hypothesis":
                self._on_hypothesis_text(event.get("text", ""))
            elif kind == "result":
                self._on_result_text(event.get("text", ""))
            elif kind == "error":
                raise RuntimeError(event.get("error") or "Windows Speech の起動に失敗しました")
        code = self._proc.poll()
        if not self._stop.is_set() and code not in (None, 0):
            stderr = ""
            if self._proc.stderr:
                try:
                    stderr = self._proc.stderr.read().strip()
                except Exception:
                    stderr = ""
            raise RuntimeError(stderr or f"Windows Speech bridge exited: {code}")

    @staticmethod
    def _bridge_path():
        base = getattr(sys, "_MEIPASS", None)
        if base:
            path = os.path.join(base, "koemo", "native_speech_bridge.ps1")
            if os.path.isfile(path):
                return path
        return str(Path(__file__).resolve().parent / "native_speech_bridge.ps1")

    def _resolve_language(self, Language, SpeechRecognizer):
        """要求言語タグをこのOSが対応する topic 言語へ寄せる。

        system speech language は `ja`（topic/grammar も `ja`）で、`ja-JP` 指定でも
        通常は解決されるが、対応リストに無ければ system speech language へ落として
        初回 compile の不一致を避ける。
        """
        requested = self.language or "ja-JP"
        try:
            supported = [l.language_tag for l in SpeechRecognizer.supported_topic_languages]
        except Exception:
            supported = []
        if not supported or requested in supported:
            return Language(requested)
        primary = requested.split("-")[0].lower()
        for tag in supported:
            if tag.lower() == primary or tag.lower().split("-")[0] == primary:
                return Language(tag)
        try:
            return SpeechRecognizer.system_speech_language
        except Exception:
            return Language(requested)

    def _on_state(self, state):
        name = (getattr(state, "name", "") or "").upper()
        if name in ("SPEECH_DETECTED", "SOUND_STARTED"):
            self._emit_speech_detected()

    def _emit_speech_detected(self):
        # エンジンが発話を検知した瞬間（最初の文字仮説より早い）にライブUIを動かす。
        # 実テキスト（仮説/確定）が既にあるなら上書きしない。
        if self._hypothesis or self._final:
            return
        self._speech_detected = True
        if self.on_update:
            self.on_update(f"{self.label}: ...")

    def _on_hypothesis_text(self, hypothesis):
        hypothesis = normalize_native_text((hypothesis or "").strip())
        if not hypothesis:
            return
        self._hypothesis = hypothesis
        self._emit()

    def _on_result_text(self, text):
        text = normalize_native_text((text or "").strip())
        if not text:
            return
        now = time.time() - self._started
        ev = LiveEvent(max(0.0, now - 1.0), now, self.label, text, "final")
        with self._rows_lock:
            self._final.append(ev)
        self._hypothesis = ""
        self._emit()

    def _emit_status(self, text):
        if self.on_update:
            self.on_update(text)

    def _emit(self):
        with self._rows_lock:
            events = list(self._final[-12:])
        if self._hypothesis:
            now = time.time() - self._started
            events.append(LiveEvent(now, now, self.label, self._hypothesis, "provisional"))
        text = format_live_events(events)
        if text and self.on_update:
            self.on_update(text)


class MicActivityPreview:
    """マイク入力の立ち上がりを即時表示する低遅延プレースホルダ。"""

    def __init__(self, recorder, label=MIC_LABEL, on_update=None,
                 min_audio_sec=0.15, silence_rms=0.01, poll_sec=0.05):
        self.recorder = recorder
        self.label = label
        self.on_update = on_update
        self.min_audio_sec = min_audio_sec
        self.silence_rms = silence_rms
        self.poll_sec = poll_sec
        self._stop = threading.Event()
        self._thread = None
        self._emitted = False
        self._started = time.time()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout=0.2):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None

    def events(self):
        if not self._emitted:
            return []
        now = time.time() - self._started
        return [LiveEvent(max(0.0, now - 0.1), now, self.label, "...", "provisional")]

    def final_rows(self):
        return []

    def _run(self):
        while not self._stop.is_set() and not self._emitted:
            try:
                if self.recorder.captured_seconds(self.label) >= self.min_audio_sec:
                    audio = self.recorder.snapshot(self.label, max_seconds=0.25)
                    if len(audio):
                        rms = float(np.sqrt(np.mean(np.square(audio))))
                        if rms >= self.silence_rms:
                            self._emitted = True
                            if self.on_update:
                                self.on_update(f"{self.label}: ...")
                            return
            except Exception:
                return
            self._stop.wait(self.poll_sec)


class LiveTranscriber:
    """ローリング窓で末尾だけ再計算し、録音中の字幕プレビューを返す。"""

    def __init__(self, recorder, transcriber, language="ja", on_update=None,
                 on_error=None, interval_sec=1.2, window_sec=8.0,
                 stable_margin_sec=1.5, min_audio_sec=0.8, silence_rms=0.001):
        self.recorder = recorder
        self.transcriber = transcriber
        self.language = language
        self.on_update = on_update
        self.on_error = on_error
        self.interval_sec = interval_sec
        self.window_sec = window_sec
        self.stable_margin_sec = stable_margin_sec
        self.min_audio_sec = min_audio_sec
        self.silence_rms = silence_rms
        self._stop = threading.Event()
        self._thread = None
        self._rows_lock = threading.Lock()
        self._committed = []
        self._committed_until = {}
        self._latest_rows = []
        self._last_text = ""
        self._active_labels = []

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout=1.0):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None

    def final_rows(self):
        return [(ev.start, ev.label, ev.text) for ev in self.events() if ev.kind == "final"]

    def events(self):
        with self._rows_lock:
            rows = list(self._latest_rows or self._committed)
        return [
            LiveEvent(st, en, label, text, "final")
            for (st, en, label, text) in self._dedupe(rows)
        ]

    def _active_audio_labels(self):
        candidates = []
        for label in (SYS_LABEL, MIC_LABEL):
            if self.recorder.captured_seconds(label) < self.min_audio_sec:
                continue
            audio = self.recorder.snapshot(label, max_seconds=3.0)
            if len(audio) == 0:
                continue
            rms = float(np.sqrt(np.mean(np.square(audio))))
            if rms >= self.silence_rms:
                candidates.append((label, rms))
        if candidates:
            candidates.sort(key=lambda item: (item[0] == SYS_LABEL, item[1]), reverse=True)
            levels = dict(candidates)
            if SYS_LABEL in levels and MIC_LABEL in levels:
                if levels[SYS_LABEL] >= levels[MIC_LABEL] * 0.75:
                    self._active_labels = [SYS_LABEL]
                    return self._active_labels
                if levels[MIC_LABEL] >= levels[SYS_LABEL] * 1.6:
                    self._active_labels = [MIC_LABEL]
                    return self._active_labels
            self._active_labels = [label for label, _rms in candidates]
            return self._active_labels
        return []

    def _run(self):
        while not self._stop.is_set():
            started = time.time()
            try:
                self._transcribe_once()
            except Exception as e:
                if self.on_error and not self._stop.is_set():
                    self.on_error(str(e))
            elapsed = time.time() - started
            wait = max(0.5, self.interval_sec - elapsed)
            self._stop.wait(wait)

    def _transcribe_once(self, force=False):
        labels = self._active_audio_labels()
        if not labels:
            return

        all_rows = []
        for label in labels:
            total_sec = self.recorder.captured_seconds(label)
            if total_sec < self.min_audio_sec and not force:
                continue

            audio = self.recorder.snapshot(label, max_seconds=self.window_sec)
            if len(audio) < int(self.min_audio_sec * self.recorder.SR) and not force:
                continue

            window_start = max(0.0, total_sec - (len(audio) / float(self.recorder.SR)))
            segs = self.transcriber.transcribe_segments(
                audio,
                language=self.language,
                vad_filter=False,
            )
            rows = []
            for start, end, text in segs:
                if text:
                    rows.append((window_start + start, window_start + end, label, text))

            stable_cutoff = max(0.0, total_sec - self.stable_margin_sec)
            last_until = self._committed_until.get(label, 0.0)
            for row in rows:
                if row[1] <= stable_cutoff and row[1] > last_until + 0.05:
                    self._committed.append(row)
                    last_until = max(last_until, row[1])
            self._committed_until[label] = last_until

            preview = [row for row in rows if row[1] > last_until + 0.05]
            all_rows.extend(preview)

        visible_rows = self._dedupe(self._committed + all_rows)
        with self._rows_lock:
            self._latest_rows = visible_rows

        text = self._format(visible_rows)
        if text and text != self._last_text:
            self._last_text = text
            if self.on_update:
                self.on_update(text)

    @staticmethod
    def _format(rows):
        return "\n".join(f"{label}: {text}" for (_st, _en, label, text) in rows)

    @staticmethod
    def _dedupe(rows):
        out = []
        for row in sorted(rows, key=lambda r: (r[0], r[2], r[3])):
            if out and out[-1][2] == row[2] and out[-1][3] == row[3]:
                if row[1] > out[-1][1]:
                    out[-1] = row
                continue
            out.append(row)
        return out
