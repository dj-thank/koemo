"""Koemo アプリ本体（PySide6）— トレイ常駐・ホットキー録音・処理オーケストレーション。"""
import os
import sys
import gc
import re
import time
import threading
from datetime import datetime
from pathlib import Path

from PySide6.QtWidgets import (QApplication, QSystemTrayIcon, QMenu, QWidget,
                               QVBoxLayout, QLabel, QMessageBox, QFileDialog)
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor, QAction, QFont
from PySide6.QtCore import Qt, QObject, Signal, Slot, QTimer

import keyboard
import numpy as np

from .config import load_config, save_config, MIC_LABEL, SYS_LABEL
from .audio import DualRecorder
from .detect import MeetingWatcher
from .live import LiveTranscriber, MicActivityPreview, NativeWindowsLiveTranscriber
from .transcribe import Transcriber, merge_rows
from .summarize import Summarizer
from .native_speech import transcribe_wav_events
from . import calendar_hint, diarize, library, readiness
from .ui_live import LiveWindow
from .ui_library import LibraryWindow
from .ui_results import ResultsWindow
from .ui_settings import SettingsDialog

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def write_text_with_fallback(path, text):
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path, None
    except Exception as e:
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Koemo" / "Recordings"
        fallback = base / path.name
        fallback.parent.mkdir(parents=True, exist_ok=True)
        fallback.write_text(text, encoding="utf-8")
        return fallback, f"{path}: {e}"


def make_icon(recording=False):
    asset = ASSETS_DIR / "koemo.png"
    if asset.is_file():
        pm = QPixmap(str(asset)).scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        if recording:
            p = QPainter(pm); p.setRenderHint(QPainter.Antialiasing); p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 220, 0)); p.drawEllipse(46, 4, 14, 14)
            p.end()
        return QIcon(pm)

    pm = QPixmap(64, 64); pm.fill(Qt.transparent)
    p = QPainter(pm); p.setRenderHint(QPainter.Antialiasing); p.setPen(Qt.NoPen)
    p.setBrush(QColor(220, 50, 50) if recording else QColor(100, 160, 240))
    p.drawEllipse(2, 2, 60, 60)
    p.setBrush(QColor(20, 20, 36))
    p.drawRoundedRect(24, 12, 16, 24, 8, 8)
    p.drawRect(29, 46, 6, 8)
    p.drawRect(23, 52, 18, 4)
    if recording:
        p.setBrush(QColor(255, 220, 0)); p.drawEllipse(46, 4, 14, 14)
    p.end()
    return QIcon(pm)


class Toast(QWidget):
    """画面右下に出る小さな通知。"""
    def __init__(self):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setStyleSheet("background:#1a1a2e;border:1px solid #3a3a5c;border-radius:8px;")
        lay = QVBoxLayout(self); lay.setContentsMargins(16, 14, 16, 14)
        self._lbl = QLabel("")
        self._lbl.setAlignment(Qt.AlignCenter)
        self._lbl.setWordWrap(True)
        self._lbl.setStyleSheet("color:#e0e0ff;border:none;")
        self._lbl.setFont(QFont("Yu Gothic UI", 11))
        lay.addWidget(self._lbl)
        self.resize(300, 92)

    def show_msg(self, text):
        self._lbl.setText(text)
        g = QApplication.primaryScreen().availableGeometry()
        self.move(g.right() - self.width() - 18, g.bottom() - self.height() - 18)
        self.show()

    def hide_toast(self):
        self.hide()


class _Bridge(QObject):
    """別スレッド→GUIスレッドへ通知するシグナル束。"""
    toggle       = Signal()
    toast        = Signal(str)
    toast_close  = Signal()
    live         = Signal(str)
    meeting      = Signal(str)
    results      = Signal(object)
    error        = Signal(str)


class KoemoApp(QObject):
    def __init__(self, app):
        super().__init__()
        self._app = app
        self.cfg = load_config()
        self.recorder = DualRecorder(self.cfg)
        self.transcriber = None
        self.live_model = None
        if not self.cfg.get("native_only_transcription", False):
            self.transcriber = Transcriber(self.cfg.get("whisper_model", "large-v3-turbo"),
                                           self.cfg.get("cpu_threads", 2),
                                           self.cfg.get("idle_unload_sec", 300),
                                           self.cfg.get("keep_warm", False))
            self.live_model = Transcriber(self.cfg.get("live_whisper_model", "small"),
                                          self.cfg.get("cpu_threads", 2),
                                          self.cfg.get("idle_unload_sec", 300),
                                          self.cfg.get("keep_warm", False))
        self.summarizer = Summarizer(self.cfg.get("summary_model_dir", ""),
                                     self.cfg.get("idle_unload_sec", 300),
                                     self.cfg.get("keep_warm", False),
                                     self.cfg)
        self.recording = False
        self.processing = False
        self._results = None
        self._settings = None
        self._library = None
        self._live_window = None
        self._live_transcriber = None
        self._live_activity = None
        self._last_live_rows = []
        self._native_fell_back = False
        self._final_model_state = "idle"
        self._meeting_watcher = None
        self._pending_cfg = None
        self._test_command_file = os.environ.get("KOEMO_TEST_COMMAND_FILE", "")
        self._test_command_seen = ""
        self._test_command_timer = None

        self._toast = Toast()
        self._timer = QTimer(self); self._timer.setInterval(1000); self._timer.timeout.connect(self._tick)

        self.sig = _Bridge()
        self.sig.toggle.connect(self._on_toggle)
        self.sig.toast.connect(self._toast.show_msg)
        self.sig.toast_close.connect(self._toast.hide_toast)
        self.sig.live.connect(self._update_live)
        self.sig.meeting.connect(self._on_meeting_detected)
        self.sig.results.connect(self._open_results)
        self.sig.error.connect(self._show_error)

        self._build_tray()

    # ── 起動 ──
    def start(self):
        self._register_hotkey()
        self._start_meeting_watcher()
        self._start_test_command_watcher()
        self._preload_transcriber()
        threading.Thread(target=self._idle_watcher, daemon=True).start()
        hk = self.cfg.get("hotkey", "ctrl+shift+r").upper()
        config_error = self.cfg.get("_config_load_error")
        if config_error:
            self._toast.show_msg(
                "⚠  設定ファイルを読み込めませんでした\n"
                f"既定値で起動します。退避先: {self.cfg.get('_config_backup', '')}"
            )
            QTimer.singleShot(8500, self._toast.hide_toast)
        else:
            self._toast.show_msg(f"✅  Koemo 起動\n{hk} で録音開始（マイク＋システム音声）")
            QTimer.singleShot(4500, self._toast.hide_toast)
            QTimer.singleShot(5200, self._show_first_run_readiness)

    # ── トレイ ──
    def _build_tray(self):
        self._tray = QSystemTrayIcon(make_icon(False))
        self._tray.setToolTip("Koemo — クリックで録音")
        menu = QMenu()
        a_rec = QAction("🎙  録音開始 / 停止", self); a_rec.triggered.connect(self._on_toggle)
        a_imp = QAction("📂  音声ファイルを取込", self); a_imp.triggered.connect(self._import_audio)
        a_lib = QAction("📚  履歴", self); a_lib.triggered.connect(self._open_library)
        a_fld = QAction("📁  録音フォルダを開く", self); a_fld.triggered.connect(self._open_folder)
        a_set = QAction("⚙  設定", self); a_set.triggered.connect(self._open_settings)
        a_qt  = QAction("✕  終了", self); a_qt.triggered.connect(self._quit)
        menu.addAction(a_rec); menu.addAction(a_imp); menu.addSeparator()
        menu.addAction(a_lib); menu.addAction(a_fld); menu.addAction(a_set); menu.addSeparator(); menu.addAction(a_qt)
        self._tray.setContextMenu(menu)
        self._tray.activated.connect(self._tray_activated)
        self._tray.show()

    def _tray_activated(self, reason):
        if reason == QSystemTrayIcon.Trigger:   # 左クリック
            self._on_toggle()

    def _set_tray_recording(self, rec):
        self._tray.setIcon(make_icon(rec))

    # ── ホットキー ──
    def _register_hotkey(self):
        try:
            try:
                keyboard.unhook_all_hotkeys()
            except Exception:
                pass
            keyboard.add_hotkey(self.cfg.get("hotkey", "ctrl+shift+r"),
                                lambda: self.sig.toggle.emit())
        except Exception as e:
            print(f"[hotkey] {e}")

    def _start_meeting_watcher(self):
        self._stop_meeting_watcher()
        if not self.cfg.get("enable_meeting_detection", True):
            return
        self._meeting_watcher = MeetingWatcher(lambda name: self.sig.meeting.emit(name))
        self._meeting_watcher.start()

    def _stop_meeting_watcher(self):
        if self._meeting_watcher:
            self._meeting_watcher.stop()
            self._meeting_watcher = None

    def _show_first_run_readiness(self):
        if self.recording or self.processing:
            return
        notice = readiness.first_run_notice(self.cfg)
        if not notice:
            return
        self._toast.show_msg("⚠  " + notice)
        QTimer.singleShot(9000, self._toast.hide_toast)

    def _start_test_command_watcher(self):
        """Integration test用。通常起動では無効で、環境変数指定時だけfile commandを読む。"""
        if not self._test_command_file:
            return
        self._test_command_timer = QTimer(self)
        self._test_command_timer.setInterval(200)
        self._test_command_timer.timeout.connect(self._poll_test_command)
        self._test_command_timer.start()

    def _poll_test_command(self):
        try:
            path = Path(self._test_command_file)
            if not path.is_file():
                return
            text = path.read_text(encoding="utf-8").strip()
        except Exception:
            return
        if not text or text == self._test_command_seen:
            return
        self._test_command_seen = text
        command = text.split(":", 1)[0].strip().lower()
        if command == "toggle":
            self.sig.toggle.emit()
        elif command == "quit":
            self._quit()

    # ── 録音トグル ──
    @Slot()
    def _on_toggle(self):
        if self.processing:
            return
        if self.recording:
            self._stop()
        else:
            self._start()

    def _start(self):
        if not (self.cfg.get("record_mic", True) or self.cfg.get("record_system", True)):
            self.sig.toast.emit("⚠️  録音対象がありません")
            self.sig.error.emit("録音対象がありません。設定でマイクまたはシステム音声の少なくとも一方を有効にしてください。")
            return
        self.recording = True
        self._last_live_rows = []
        if (not self.cfg.get("native_only_transcription", False)
                and self._final_model_state != "ready"
                and self.cfg.get("show_model_ready_status", True)):
            self.sig.toast.emit("🎯  高精度モデル準備中です。初回だけ停止後処理が遅くなることがあります")
        self.recorder.start()
        self._start_live()
        self._set_tray_recording(True)
        self._timer.start()
        self._tick()

    def _tick(self):
        d = self.recorder.elapsed(); m, s = divmod(d, 60)
        hk = self.cfg.get("hotkey", "ctrl+shift+r").upper()
        self._toast.show_msg(f"🔴  録音中  {m:02d}:{s:02d}\n停止: {hk}")

    def _stop(self):
        self.recording = False
        self.processing = True
        self._timer.stop()
        self._stop_live()
        self._set_tray_recording(False)
        self._toast.show_msg("⏹  停止 — 処理中...")
        threading.Thread(target=self._process, daemon=True).start()

    def _resolve_live_backend(self):
        """live_backend="auto" を実機に合わせて解決する。GPUありなら日本語精度の
        高い whisper_rolling、無ければOS依存だが軽い native_windows を選ぶ。
        明示指定(native_windows/whisper_rolling/off)はそのまま尊重する。"""
        backend = self.cfg.get("live_backend", "auto")
        if backend == "auto":
            from .gpu import gpu_ok
            return "whisper_rolling" if gpu_ok() else "native_windows"
        return backend

    def _effective_live_backend(self, backend):
        """録音設定に合わないライブbackendを実録音chベースへ寄せる。"""
        if backend == "native_windows" and not self.cfg.get("record_mic", True):
            # Windows native live はOSの既定マイクを直接開く。マイク録音OFFの
            # system-only録音では、プライバシー設定を尊重して録音済みchだけを
            # 使う whisper rolling へ落とす。
            fallback = self.cfg.get("live_fallback_backend", "off")
            return "whisper_rolling" if fallback == "whisper_rolling" else "off"
        return backend

    def _start_live(self):
        backend = self._effective_live_backend(self._resolve_live_backend())
        if (not self.cfg.get("enable_live_transcription", True)) or backend == "off":
            return
        self._native_fell_back = False
        self._live_window = LiveWindow()
        self._live_window.update_text("音声認識準備中...")
        self._live_window.show_at_corner()
        if backend == "native_windows":
            self._live_activity = MicActivityPreview(
                self.recorder,
                on_update=lambda text: self.sig.live.emit(text),
            )
            self._live_activity.start()
        self._live_transcriber = self._make_live_transcriber(backend)
        if self._live_transcriber is None:
            if self._live_activity:
                self._live_activity.stop(timeout=0.2)
                self._live_activity = None
            self._live_window.update_text("ライブ文字起こしを開始できません")
            return
        self._live_transcriber.start()

    def _make_live_transcriber(self, backend):
        backend = self._effective_live_backend(backend)
        if backend == "off":
            return None

        def fallback(reason):
            fb = self.cfg.get("live_fallback_backend", "off")
            if fb == "whisper_rolling" and backend != "whisper_rolling":
                self.sig.live.emit("Windows音声認識が使えないためWhisper速報に切替")
                return self._make_live_transcriber("whisper_rolling")
            self.sig.toast.emit(f"Windows文字起こし: {reason}")
            return None

        if backend == "native_windows":
            try:
                native = NativeWindowsLiveTranscriber(
                    language=self.cfg.get("native_speech_language", "ja-JP"),
                    startup_settle_sec=float(self.cfg.get("native_speech_startup_settle_sec", 1.0)),
                    on_update=lambda text: self.sig.live.emit(text),
                    on_error=lambda msg: self.sig.toast.emit(f"Windows音声認識: {msg}"),
                    on_unavailable=lambda msg: self._fallback_live_after_native(msg),
                )
                native.start()
                return native
            except Exception as e:
                return fallback(str(e))

        if backend != "whisper_rolling":
            return fallback(f"未知のライブバックエンド: {backend}")

        if self.live_model is None:
            self.live_model = Transcriber(self.cfg.get("live_whisper_model", "small"),
                                          self.cfg.get("cpu_threads", 2),
                                          self.cfg.get("idle_unload_sec", 300),
                                          self.cfg.get("keep_warm", False))

        lang = self.cfg.get("summary_language", "ja")
        return LiveTranscriber(
            self.recorder,
            self.live_model,
            language=lang,
            on_update=lambda text: self.sig.live.emit(text),
            on_error=lambda msg: self.sig.toast.emit(f"ライブ文字起こし: {msg}"),
            interval_sec=float(self.cfg.get("live_interval_sec", 1.2)),
            window_sec=float(self.cfg.get("live_window_sec", 8.0)),
            stable_margin_sec=float(self.cfg.get("live_stable_margin_sec", 1.5)),
            min_audio_sec=float(self.cfg.get("live_min_audio_sec", 0.8)),
        )

    def _fallback_live_after_native(self, reason):
        if self._native_fell_back:
            return
        if not self.recording or self.cfg.get("live_fallback_backend") != "whisper_rolling":
            self.sig.toast.emit(f"Windows文字起こし: {reason}")
            return
        self._native_fell_back = True
        self.sig.live.emit("Windows音声認識が使えないためWhisper速報に切替")
        self.sig.toast.emit(f"Windows音声認識: {reason}")
        fallback = self._make_live_transcriber("whisper_rolling")
        if fallback is None:
            return
        self._live_transcriber = fallback
        fallback.start()

    def _stop_live(self):
        if self._live_activity:
            self._live_activity.stop(timeout=0.2)
            self._live_activity = None
        if self._live_transcriber:
            lt = self._live_transcriber
            lt.stop(timeout=0.2)
            self._last_live_rows = lt.final_rows()
            self._live_transcriber = None
        if self._live_window:
            self._live_window.close()
            self._live_window = None

    def _process(self):
        rec = None
        try:
            cfg = dict(self.cfg)
            rec = self.recorder.stop()
            if rec is None:
                self.sig.toast_close.emit(); return
            if not rec.get("channels"):
                errors = list((rec.get("capture_errors") or {}).values())
                if errors:
                    detail = "\n".join(f"- {e}" for e in errors)
                    raise RuntimeError(f"音声取得に失敗しました:\n{detail}")
                raise RuntimeError("録音音声が取得できませんでした。マイク/システム音声の設定を確認してください。")
            lang = cfg.get("summary_language", "ja")

            live_rows = list(self._last_live_rows or [])
            if cfg.get("native_only_transcription", False):
                self.sig.toast.emit("📝  Windows文字起こし結果を整理中...")
                rows = self._native_final_rows(rec, live_rows, cfg)
                transcript_md = merge_rows(rows)
                if not transcript_md.strip():
                    transcript_md = "（Windows音声認識の確定結果がありません。マイク入力、音声認識言語、Windowsの音声認識設定を確認してください）"
            elif cfg.get("use_live_transcript_on_stop", False) and live_rows:
                self.sig.toast.emit("📝  ライブ文字起こし結果を整理中...")
                transcript_md = merge_rows(live_rows)
            else:
                do_diar = (not cfg.get("fast_summary", True)
                           and cfg.get("enable_diarization", True)
                           and diarize.available())
                rows = []
                for label, audio in self._active_final_channels(rec["channels"], cfg):
                    if self.transcriber is None:
                        raise RuntimeError("Whisper transcriber is disabled")
                    self.sig.toast.emit(f"📝  高精度文字起こし中（{label}）...")
                    segs = self.transcriber.transcribe_segments(
                        audio, language=lang,
                        on_progress=lambda m, l=label: self.sig.toast.emit(f"📝  {l}: {m}"))
                    if do_diar and label == SYS_LABEL and len(segs) > 1:
                        self.sig.toast.emit("🗣  話者を識別中...")
                        rows.extend(diarize.assign_speakers(segs, diarize.diarize(audio), label))
                    else:
                        rows.extend((st, label, txt) for (st, en, txt) in segs)
                audio = None
                rows = self._filter_final_rows(rows)
                transcript_md = merge_rows(rows)

            # ここから先は録音音声そのものを使わないため、長尺spoolを早めに閉じて削除する。
            self._cleanup_recording_temp_files(rec)

            try:
                if cfg.get("fast_summary", True):
                    self.sig.toast.emit("⚡  即時要約を作成中...")
                    title, summary_md = self.summarizer.fast_summarize(transcript_md, language=lang)
                else:
                    self._release_transcription_models_for_summary(cfg)
                    title, summary_md = self.summarizer.summarize(
                        transcript_md, language=lang,
                        on_progress=lambda m: self.sig.toast.emit(f"🤖  {m}"))
            except Exception as se:
                title = "会議メモ"
                summary_md = f"⚠️  要約の生成に失敗しました:\n{se}\n\n（文字起こしは下に保存されています）"
            title = calendar_hint.apply_title_hint(cfg, title)

            save_dir = Path(cfg["save_dir"]); ts = rec["ts"]
            # 録音スレッドの取得失敗を surface（黙って一部chが欠落しないように）。
            capture_errors = list((rec.get("capture_errors") or {}).values())
            if capture_errors:
                self.sig.toast.emit("⚠️  一部の音声取得に失敗しましたが、処理は継続します")
                capture_md = "\n".join(f"- {w}" for w in capture_errors)
                summary_md = f"{summary_md}\n\n## 録音警告\n{capture_md}"
            warnings = list((rec.get("write_errors") or {}).values())
            if warnings:
                self.sig.toast.emit("⚠️  録音WAV保存に失敗しましたが、処理は継続します")
                warning_md = "\n".join(f"- {w}" for w in warnings)
                summary_md = f"{summary_md}\n\n## 保存警告\n{warning_md}"
            temp_warnings = list(rec.get("temp_cleanup_errors") or [])
            if temp_warnings:
                self.sig.toast.emit("⚠️  一時録音ファイルの削除に失敗しました")
                temp_warning_md = "\n".join(f"- {w}" for w in temp_warnings)
                summary_md = f"{summary_md}\n\n## 一時ファイル警告\n{temp_warning_md}"
            doc = f"# {title}\n\n{summary_md}\n\n---\n\n## 文字起こし\n\n{transcript_md}\n"
            doc_path, doc_error = write_text_with_fallback(save_dir / f"summary_{ts}.md", doc)
            if doc_error:
                summary_md = f"{summary_md}\n\n## 保存警告\n- {doc_error}\n- 保存先を {doc_path.parent} に切り替えました"
                doc = f"# {title}\n\n{summary_md}\n\n---\n\n## 文字起こし\n\n{transcript_md}\n"
                doc_path.write_text(doc, encoding="utf-8")
            save_dir = doc_path.parent
            self._add_library(title, summary_md, transcript_md, save_dir, rec["duration"], ts)

            self.sig.toast_close.emit()
            self.sig.results.emit({"title": title, "summary": summary_md,
                                   "transcript": transcript_md, "dir": save_dir,
                                   "dur": rec["duration"]})
            gc.collect()
        except Exception as e:
            self.sig.toast_close.emit()
            self.sig.error.emit(str(e))
        finally:
            self._cleanup_recording_temp_files(rec)
            self.processing = False
            self._apply_pending_cfg_if_idle()

    def _release_transcription_models_for_summary(self, cfg=None):
        """LLM要約前にWhisperモデルを退避し、8GB VRAM環境で3モデル同時常駐を避ける。"""
        cfg = cfg or self.cfg
        if cfg.get("keep_warm", False):
            return
        for model in (self.live_model, self.transcriber):
            unload = getattr(model, "unload", None)
            if unload:
                unload()

    @staticmethod
    def _cleanup_recording_temp_files(rec):
        """録音中のディスクスプールを、処理完了後に削除する。"""
        if not rec:
            return
        for audio in list((rec.get("channels") or {}).values()):
            KoemoApp._close_array_backing(audio)
        rec["channels"] = {}
        gc.collect()
        remaining = []
        errors = rec.setdefault("temp_cleanup_errors", [])
        for raw_path in list(rec.get("temp_files") or []):
            path = Path(raw_path)
            err = None
            deleted = False
            for _attempt in range(5):
                try:
                    if not path.exists():
                        deleted = True
                        break
                    path.unlink()
                    deleted = True
                    err = None
                    break
                except Exception as e:
                    err = e
                    gc.collect()
                    time.sleep(0.1)
            if not deleted:
                remaining.append(str(path))
                if err is not None:
                    msg = f"{path}: {err}"
                    if msg not in errors:
                        errors.append(msg)
        rec["temp_files"] = remaining

    @staticmethod
    def _close_array_backing(audio):
        """numpy.memmap とその base chain の mmap handle を明示的に閉じる。"""
        seen = set()
        cur = audio
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            if isinstance(cur, np.memmap):
                mmap_obj = getattr(cur, "_mmap", None)
                if mmap_obj is not None:
                    try:
                        mmap_obj.close()
                    except Exception:
                        pass
            cur = getattr(cur, "base", None)

    def _import_audio(self):
        if self.processing:
            return
        path, _ = QFileDialog.getOpenFileName(
            None, "音声ファイルを取込", str(Path(self.cfg["save_dir"])),
            "音声 (*.wav *.mp3 *.m4a *.flac *.ogg *.mp4 *.aac);;すべて (*.*)")
        if not path:
            return
        self.processing = True
        self._toast.show_msg("📂  取込処理中...")
        threading.Thread(target=self._process_file, args=(path,), daemon=True).start()

    def _process_file(self, path):
        try:
            cfg = dict(self.cfg)
            lang = cfg.get("summary_language", "ja")
            name = Path(path).stem
            self.sig.toast.emit("📝  文字起こし中...")
            if cfg.get("native_only_transcription", False):
                if Path(path).suffix.lower() != ".wav":
                    raise RuntimeError("Windows純正のみモードの音声取込は WAV のみ対応です。")
                events = transcribe_wav_events(
                    path,
                    language=cfg.get("native_speech_language", "ja-JP"),
                    timeout_sec=90,
                )
                rows = []
                for i, event in enumerate(events):
                    text = event.get("text") or ""
                    conf = float(event.get("confidence") or 0.0)
                    if text and self._native_rows_quality([(float(i), MIC_LABEL, text)], conf) >= 4.0:
                        rows.append((float(i), MIC_LABEL, text))
                transcript_md = merge_rows(rows)
                segs = [(float(i), float(i + 1), row[2]) for i, row in enumerate(rows)]
                if not transcript_md.strip():
                    transcript_md = "（Windows音声認識の信頼度が低いため、正式な文字起こしとして採用しませんでした）"
            else:
                segs = self.transcriber.transcribe_segments(
                    path, language=lang,
                    on_progress=lambda m: self.sig.toast.emit(f"📝  {m}"))
                transcript_md = "\n".join(t for (_s, _e, t) in segs)
            try:
                if cfg.get("fast_summary", True):
                    title, summary_md = self.summarizer.fast_summarize(transcript_md, language=lang)
                else:
                    self._release_transcription_models_for_summary(cfg)
                    title, summary_md = self.summarizer.summarize(
                        transcript_md, language=lang,
                        on_progress=lambda m: self.sig.toast.emit(f"🤖  {m}"))
            except Exception as se:
                title, summary_md = name, f"⚠️  要約の生成に失敗しました:\n{se}"
            save_dir = Path(cfg["save_dir"])
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            doc = f"# {title}\n\n{summary_md}\n\n---\n\n## 文字起こし\n\n{transcript_md}\n"
            doc_path, doc_error = write_text_with_fallback(save_dir / f"import_{ts}.md", doc)
            if doc_error:
                summary_md = f"{summary_md}\n\n## 保存警告\n- {doc_error}\n- 保存先を {doc_path.parent} に切り替えました"
                doc = f"# {title}\n\n{summary_md}\n\n---\n\n## 文字起こし\n\n{transcript_md}\n"
                doc_path.write_text(doc, encoding="utf-8")
            save_dir = doc_path.parent
            dur = int(segs[-1][1]) if segs else 0
            self._add_library(title, summary_md, transcript_md, save_dir, dur, ts)
            self.sig.toast_close.emit()
            self.sig.results.emit({"title": title, "summary": summary_md,
                                   "transcript": transcript_md, "dir": save_dir, "dur": dur})
            gc.collect()
        except Exception as e:
            self.sig.toast_close.emit()
            self.sig.error.emit(str(e))
        finally:
            self.processing = False
            self._apply_pending_cfg_if_idle()

    @Slot(object)
    def _open_results(self, r):
        chat_func = self._make_chat_func(r["transcript"])

        self._results = ResultsWindow(r["title"], r["summary"], r["transcript"],
                                      r["dir"], r["dur"], chat_func=chat_func)
        self._results.show()
        self._results.raise_()
        self._results.activateWindow()

    @Slot(str)
    def _update_live(self, text):
        if self._live_window:
            self._live_window.update_text(text)

    @Slot(str)
    def _on_meeting_detected(self, app_name):
        hk = self.cfg.get("hotkey", "ctrl+shift+r").upper()
        self._tray.showMessage("会議を検出", f"{app_name} が起動中です。録音: {hk}",
                               QSystemTrayIcon.Information, 6000)

    @Slot(str)
    def _show_error(self, msg):
        QMessageBox.critical(None, "エラー", f"処理中にエラー:\n{msg}")

    # ── メニュー ──
    def _open_folder(self):
        d = Path(self.cfg["save_dir"]); d.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(d))
        except Exception:
            pass

    def _open_settings(self):
        self._settings = SettingsDialog(self.cfg, self._on_cfg_saved)
        self._settings.show()

    def _open_library(self):
        self._library = LibraryWindow(chat_factory=self._make_chat_func)
        self._library.show()
        self._library.raise_()
        self._library.activateWindow()

    def _make_chat_func(self, transcript):
        def chat_func(question, history):
            return self.summarizer.chat(
                question, transcript, history,
                language=self.cfg.get("summary_language", "ja"))
        return chat_func

    @staticmethod
    def _add_library(title, summary_md, transcript_md, save_dir, duration, ts):
        try:
            library.add(title, summary_md, transcript_md, save_dir, duration, ts)
        except Exception as e:
            print(f"[library] {e}")

    def _on_cfg_saved(self, new_cfg):
        if self.recording or self.processing:
            self._pending_cfg = dict(new_cfg)
            self.sig.toast.emit("⚙  設定は現在の録音/処理が終わってから反映します")
            return
        self._apply_cfg(new_cfg)

    def _apply_pending_cfg_if_idle(self):
        if getattr(self, "_pending_cfg", None) and not self.recording and not self.processing:
            pending = self._pending_cfg
            self._pending_cfg = None
            self._apply_cfg(pending)

    def _apply_cfg(self, new_cfg):
        self.cfg = dict(new_cfg)
        # 録音中・処理中は recorder を差し替えない。_stop/_process/ライブスレッドが
        # 現 recorder を保持しているため、ここで入れ替えると競合する。新しい
        # recorder/設定は次回録音から有効になる（ユーザーはブロックしない）。
        self.recorder = DualRecorder(self.cfg)
        if self.cfg.get("native_only_transcription", False):
            self.transcriber = None
            self.live_model = None
        else:
            if self.transcriber is None:
                self.transcriber = Transcriber(self.cfg.get("whisper_model", "large-v3-turbo"),
                                               self.cfg.get("cpu_threads", 2),
                                               self.cfg.get("idle_unload_sec", 300),
                                               self.cfg.get("keep_warm", False))
            else:
                self.transcriber.reload(self.cfg.get("whisper_model", "large-v3-turbo"),
                                        self.cfg.get("cpu_threads", 2),
                                        self.cfg.get("idle_unload_sec", 300),
                                        self.cfg.get("keep_warm", False))
            if self.live_model is None:
                self.live_model = Transcriber(self.cfg.get("live_whisper_model", "small"),
                                              self.cfg.get("cpu_threads", 2),
                                              self.cfg.get("idle_unload_sec", 300),
                                              self.cfg.get("keep_warm", False))
            else:
                self.live_model.reload(self.cfg.get("live_whisper_model", "small"),
                                       self.cfg.get("cpu_threads", 2),
                                       self.cfg.get("idle_unload_sec", 300),
                                       self.cfg.get("keep_warm", False))
        self.summarizer.reload(self.cfg.get("summary_model_dir", ""),
                               self.cfg.get("idle_unload_sec", 300),
                               self.cfg.get("keep_warm", False),
                               self.cfg)
        self._register_hotkey()
        self._start_meeting_watcher()
        self._preload_transcriber()

    def _preload_transcriber(self):
        if self.cfg.get("native_only_transcription", False):
            self._final_model_state = "ready"
            return
        if not self.cfg.get("preload_transcriber", True):
            return
        def run():
            try:
                if self.cfg.get("preload_final_transcriber", True):
                    self._final_model_state = "loading"
                    if self.cfg.get("show_model_ready_status", True):
                        self.sig.toast.emit("🎯  高精度文字起こしモデルを準備中...")
                    self.transcriber.warmup()
                    self._final_model_state = "ready"
                    if self.cfg.get("show_model_ready_status", True):
                        self.sig.toast.emit("✅  高精度モデル準備完了")
                        threading.Timer(2.5, lambda: self.sig.toast_close.emit()).start()
                live_backend = self._effective_live_backend(self._resolve_live_backend())
                if live_backend == "whisper_rolling" and self.live_model is not None:
                    self.live_model.warmup()
            except Exception as e:
                self._final_model_state = "failed"
                msg = str(e)
                print(f"[transcriber preload] {msg}")
                if self.cfg.get("show_model_ready_status", True):
                    self.sig.toast.emit("⚠  高精度モデルを準備できませんでした\n" + msg)
        threading.Thread(target=run, daemon=True).start()

    @staticmethod
    def _finite(audio):
        """AEC後に紛れ込み得る NaN/inf を除いた有限サンプルだけを返す。"""
        if audio is None or len(audio) == 0:
            return np.zeros(0, dtype=np.float32)
        a = audio.astype(np.float32, copy=False)
        return a[np.isfinite(a)]

    @staticmethod
    def _rms(audio):
        a = KoemoApp._finite(audio)
        if a.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(a))))

    @staticmethod
    def _clip_fraction(audio, threshold=0.98):
        """フルスケール付近に張り付いたサンプル比率（音響エコー飽和の指標）。"""
        a = KoemoApp._finite(audio)
        if a.size == 0:
            return 0.0
        return float(np.mean(np.abs(a) >= threshold))

    def _mic_is_saturated_echo(self, channels, levels):
        """マイクがスピーカー音を過大入力で拾った飽和エコーかどうかを判定する。

        ループバック(システム)はデジタル取得でほぼクリップしないが、スピーカーを
        大音量で鳴らすとマイクはその音響エコーで飽和する（実測クリップ 6〜17%）。
        この歪んだマイク信号で Whisper は「ご視聴ありがとうございました」等を幻聴
        する一方、クリーンなシステムchは正しく文字起こしできる。相互相関は AEC と
        クリップで線形相関が壊れて当てにならない（実測ピーク 0.13〜0.20）ため、
        クリップ飽和の左右差を判定材料に使う: マイクだけが激しくクリップし、
        システムchがクリーンで、かつマイクが小さくないときをエコーとみなす。

        既知の許容トレードオフ: 入力ゲイン過大で本人発話が常時クリップ(>=3%)し、
        同時にクリーンなシステム音もある稀なケースでは、その本人発話chも捨てる。
        重クリップ音声は元々Whisperに不向きで、クリーンchを優先する方が安全という
        判断（クリップ飽和では音響エコーと本人大声を確実に区別できない。テキスト
        類似度は幻聴を弾けず、Whisper信頼度もこの種の幻聴では高く出やすいため）。
        全chを明示的に残したいときは final_channel_policy="all_active" を使う。
        """
        if SYS_LABEL not in levels or MIC_LABEL not in levels:
            return False
        mic_clip = self._clip_fraction(channels[MIC_LABEL])
        sys_clip = self._clip_fraction(channels[SYS_LABEL])
        return (mic_clip >= 0.03 and sys_clip <= 0.005
                and levels[MIC_LABEL] >= levels[SYS_LABEL])

    def _mic_is_low_level_system_leak(self, channels, levels):
        """AEC後に残った低レベルのスピーカー漏れを final transcript から外す。

        GPU Whisper はCPU経路より低音量の残留エコーも文字に起こしやすい。system
        ループバックが十分強く、mic がその 14% 以下かつ絶対音量も小さい時は、
        mic を発話チャンネルではなく残留漏れとして扱う。全ch保持が必要な場合は
        all_active を使う。
        """
        if SYS_LABEL not in levels or MIC_LABEL not in levels:
            return False
        sys_rms = levels[SYS_LABEL]
        mic_rms = levels[MIC_LABEL]
        if sys_rms < 0.12 or mic_rms <= 0:
            return False
        mic_clip = self._clip_fraction(channels[MIC_LABEL])
        return mic_clip <= 0.001 and mic_rms <= 0.035 and mic_rms <= (sys_rms * 0.14)

    def _active_final_channels(self, channels, cfg=None):
        cfg = cfg or self.cfg
        scored = []
        for label, audio in channels.items():
            rms = self._rms(audio)
            if rms >= 0.001:
                scored.append((label, audio, rms))
        if not scored:
            return list(channels.items())

        levels = {label: rms for label, _audio, rms in scored}
        # all_active は「有音chを全部使う」明示指定なので、エコー判定より優先して尊重する。
        if cfg.get("final_channel_policy", "auto_dedupe") == "all_active":
            return [(label, audio) for label, audio, _rms in scored]
        # マイクがスピーカー音の飽和エコーなら、音量で選ぶとエコーを正式 transcript
        # にしてしまう。クリーンなシステムループバックを優先しマイクを捨てる。
        if self._mic_is_saturated_echo(channels, levels):
            return [(label, audio) for label, audio, _rms in scored if label == SYS_LABEL]
        # AECで大半は消えても、低レベルに残ったスピーカー漏れをGPU Whisperが
        # もっともらしい別発話として拾う場合がある。system が圧倒的に強い時は
        # クリーンなループバックを優先する。
        if self._mic_is_low_level_system_leak(channels, levels):
            return [(label, audio) for label, audio, _rms in scored if label == SYS_LABEL]
        return [(label, audio) for label, audio, _rms in scored]

    def _native_final_rows(self, rec, live_rows, cfg=None):
        cfg = cfg or self.cfg
        candidates = []
        files = rec.get("files") or {}
        language = cfg.get("native_speech_language", "ja-JP")
        if live_rows:
            candidates.append(("live", list(live_rows), self._native_rows_quality(live_rows, source="live")))
        # Windows純正ASRでは音量だけで採用チャンネルを決めると、スピーカー音を
        # マイクが拾った低品質エコーを正式 transcript にしてしまう。まず有音
        # チャンネルを全部再認識し、文字列品質で採用側を決める。
        for label, _audio in self._active_native_channels(rec["channels"]):
            wav = files.get(label)
            if not wav:
                continue
            try:
                self.sig.toast.emit(f"📝  Windows純正で再認識中（{label}）...")
                events = transcribe_wav_events(wav, language=language, timeout_sec=45)
                rows = [(float(i), label, event["text"]) for i, event in enumerate(events) if event.get("text")]
                if rows:
                    avg_conf = sum(float(e.get("confidence") or 0.0) for e in events) / max(1, len(events))
                    candidates.append((label, rows, self._native_rows_quality(rows, avg_conf)))
            except Exception as e:
                self.sig.toast.emit(f"Windows再認識: {e}")
        rows = self._select_native_rows(candidates, cfg)
        if rows:
            return rows
        return list(live_rows or [])

    def _active_native_channels(self, channels):
        scored = []
        for label, audio in channels.items():
            rms = self._rms(audio)
            if rms >= 0.001:
                scored.append((label, audio, rms))
        if scored:
            return [(label, audio) for label, audio, _rms in scored]
        return list(channels.items())

    @staticmethod
    def _common_whisper_hallucination(text):
        normalized = re.sub(r"[\s。．、,.!！?？]+", "", text or "")
        return normalized in {
            "ご視聴ありがとうございました",
            "ご清聴ありがとうございました",
            "ありがとうございました",
        }

    def _filter_final_rows(self, rows):
        """他chに実内容がある時だけ、無音/エコー由来の典型Whisper幻聴を落とす。"""
        rows = list(rows or [])
        if len({label for _st, label, _text in rows}) < 2:
            return rows
        good_rows = [row for row in rows if not self._common_whisper_hallucination(row[2])]
        return good_rows or rows

    @staticmethod
    def _native_rows_quality(rows, avg_conf=0.0, source="file"):
        text = "。".join(t for _st, _label, t in rows if t)
        if not text:
            return -100.0
        jp = len(re.findall(r"[ぁ-んァ-ン一-龯]", text))
        latin = len(re.findall(r"[A-Za-z]", text))
        kana_noise = len(re.findall(r"[ッっー]{2,}|[ァ-ン]{1,2}(?=[ァ-ン]{1,2})", text))
        domain_hits = sum(1 for term in ("コエモ", "ライブ", "文字起こし", "停止後", "Windows", "音声認識")
                          if term in text)
        length_score = min(len(text), 120) / 8.0
        jp_ratio = jp / max(1, len(text))
        latin_penalty = latin * 2.5
        noise_penalty = kana_noise * 1.2
        short_penalty = 12.0 if len(text) < 6 else 0.0
        confidence_score = max(0.0, min(1.0, float(avg_conf or 0.0))) * 35.0
        low_conf_penalty = 28.0 if source != "live" and avg_conf and avg_conf < 0.18 else 0.0
        live_bonus = 4.0 if source == "live" else 0.0
        return (length_score + (jp_ratio * 20.0) + (domain_hits * 8.0)
                + confidence_score + live_bonus
                - latin_penalty - noise_penalty - short_penalty - low_conf_penalty)

    @staticmethod
    def _similar_native_text(a, b):
        ca = set(re.findall(r"[ぁ-んァ-ン一-龯A-Za-z0-9]+", a))
        cb = set(re.findall(r"[ぁ-んァ-ン一-龯A-Za-z0-9]+", b))
        if not ca or not cb:
            return False
        return len(ca & cb) / max(1, min(len(ca), len(cb))) >= 0.65

    def _select_native_rows(self, candidates, cfg=None):
        cfg = cfg or self.cfg
        candidates = [c for c in candidates if c[1] and c[2] >= 4.0]
        if not candidates:
            return []
        if cfg.get("final_channel_policy", "auto_dedupe") == "all_active":
            out = []
            for _label, rows, _score in candidates:
                out.extend(rows)
            return sorted(out, key=lambda row: (row[0], row[1]))

        candidates.sort(key=lambda item: item[2], reverse=True)
        best_label, best_rows, best_score = candidates[0]
        best_text = " ".join(row[2] for row in best_rows)
        out = list(best_rows)
        for label, rows, score in candidates[1:]:
            text = " ".join(row[2] for row in rows)
            if score < max(6.0, best_score * 0.55):
                continue
            if self._similar_native_text(best_text, text):
                continue
            out.extend(rows)
        return sorted(out, key=lambda row: (row[0], row[1]))

    def _quit(self):
        self._stop_live()
        self._stop_meeting_watcher()
        if self.recording:
            self._cleanup_recording_temp_files(self.recorder.stop())
        self._tray.hide()
        self._app.quit()

    def _idle_watcher(self):
        while True:
            time.sleep(60)
            if not self.recording and not self.processing:
                if self.transcriber:
                    self.transcriber.maybe_unload()
                if self.live_model:
                    self.live_model.maybe_unload()
                self.summarizer.maybe_unload()
                gc.collect()


def main():
    try:
        from .log import setup_logging
        log_file = setup_logging()
        import logging
        logging.getLogger("koemo").info("Koemo starting (log=%s)", log_file)
    except Exception:
        pass  # ログ基盤の失敗でアプリ起動を止めない
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    QApplication.setApplicationName("Koemo")
    app.setWindowIcon(make_icon(False))
    koemo = KoemoApp(app)
    koemo.start()
    run_event_loop = app.exec          # Qtイベントループ（bound method を名前束縛）
    sys.exit(run_event_loop())
