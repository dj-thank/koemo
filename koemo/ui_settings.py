"""設定ダイアログ。"""
from copy import deepcopy

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
                               QComboBox, QLineEdit, QPushButton, QFileDialog, QWidget,
                               QTextEdit, QScrollArea, QMessageBox)
from PySide6.QtCore import Qt

from .config import RECORDINGS_DIR, save_config
from .audio import list_devices

_LBL = "color:#7777aa;font-size:12px;"
_FIELD = "background:#1a1a2e;color:#d0d0ee;border:none;border-radius:6px;padding:6px;"


class SettingsDialog(QDialog):
    def __init__(self, cfg, on_save, parent=None):
        super().__init__(parent)
        # 録音/処理中に設定画面を開いても、保存前の編集で app.cfg を直接変えない。
        self._cfg = deepcopy(cfg)
        self._on_save = on_save
        self.setWindowTitle("Koemo — 設定")
        self.resize(620, 820)
        self.setStyleSheet("background:#0d0d1a;color:#d0d0ee;")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border:none;background:#0d0d1a;")
        body = QWidget()
        body.setStyleSheet("background:#0d0d1a;color:#d0d0ee;")
        root = QVBoxLayout(body)
        root.setContentsMargins(28, 22, 28, 18)
        root.setSpacing(6)
        scroll.setWidget(body)
        outer.addWidget(scroll)

        title = QLabel("⚙  設定")
        title.setStyleSheet("color:#a0c4ff;font-size:18px;font-weight:bold;")
        root.addWidget(title)

        def lbl(t):
            w = QLabel(t); w.setStyleSheet(_LBL); root.addWidget(w)

        # 録音設定
        lbl("録音する音声")
        self.cb_mic = QCheckBox("🎤 マイク（あなたの声）")
        self.cb_sys = QCheckBox("🔊 システム音声（相手の声・会議音声）")
        self.cb_aec = QCheckBox("🪄 エコー除去（スピーカー再生時のマイク回り込みを抑制）")
        self.cb_diar = QCheckBox("🗣 話者分離（相手側を 相手1/相手2 に区別）")
        self.cb_live = QCheckBox("📝 ライブ文字起こし（録音中にプレビュー表示）")
        self.cb_detect = QCheckBox("📅 会議アプリ検出（Zoom/Teams等を通知）")
        self.cb_calendar_title = QCheckBox("📆 予定タイトルを会議名に使う（ICS/Outlook）")
        self.cb_mic.setChecked(cfg.get("record_mic", True))
        self.cb_sys.setChecked(cfg.get("record_system", True))
        self.cb_aec.setChecked(cfg.get("enable_aec", True))
        self.cb_diar.setChecked(cfg.get("enable_diarization", True))
        self.cb_live.setChecked(cfg.get("enable_live_transcription", True))
        self.cb_detect.setChecked(cfg.get("enable_meeting_detection", True))
        self.cb_calendar_title.setChecked(cfg.get("enable_calendar_title_hint", False))
        for c in (self.cb_mic, self.cb_sys, self.cb_aec, self.cb_diar, self.cb_live, self.cb_detect, self.cb_calendar_title):
            c.setStyleSheet("color:#d0d0ee;"); root.addWidget(c)

        mics, spks = list_devices()
        lbl("マイク（空欄=既定）")
        self.co_mic = QComboBox(); self.co_mic.setStyleSheet(_FIELD)
        self.co_mic.addItem("（既定）", ""); [self.co_mic.addItem(m, m) for m in mics]
        self._select(self.co_mic, cfg.get("mic_name", ""))
        root.addWidget(self.co_mic)

        lbl("スピーカー（システム音声の取得元 / 空欄=既定）")
        self.co_spk = QComboBox(); self.co_spk.setStyleSheet(_FIELD)
        self.co_spk.addItem("（既定）", ""); [self.co_spk.addItem(s, s) for s in spks]
        self._select(self.co_spk, cfg.get("speaker_name", ""))
        root.addWidget(self.co_spk)

        lbl("最終文字起こしモデル（large-v3-turbo=推奨・高速高精度 / small=軽量 / large-v3=最高精度）")
        self.co_model = QComboBox(); self.co_model.setStyleSheet(_FIELD)
        for m in ["tiny", "base", "small", "medium", "large-v3-turbo", "large-v3"]:
            self.co_model.addItem(m, m)
        self._select(self.co_model, cfg.get("whisper_model", "large-v3-turbo"))
        root.addWidget(self.co_model)

        lbl("停止後のチャンネル選択")
        self.co_final_channel_policy = QComboBox(); self.co_final_channel_policy.setStyleSheet(_FIELD)
        self.co_final_channel_policy.addItem("自動で重複/マイクエコーを抑える（推奨）", "auto_dedupe")
        self.co_final_channel_policy.addItem("有音チャンネルをすべて残す", "all_active")
        self._select(self.co_final_channel_policy, cfg.get("final_channel_policy", "auto_dedupe"))
        root.addWidget(self.co_final_channel_policy)

        lbl("ライブ文字起こしバックエンド（Windows純正=低遅延）")
        self.co_live_backend = QComboBox(); self.co_live_backend.setStyleSheet(_FIELD)
        self.co_live_backend.addItem("自動（GPU=Whisper高精度 / CPU=Windows純正）推奨", "auto")
        self.co_live_backend.addItem("Windows純正（低遅延）", "native_windows")
        self.co_live_backend.addItem("Whisper速報", "whisper_rolling")
        self.co_live_backend.addItem("オフ", "off")
        self._select(self.co_live_backend, cfg.get("live_backend", "auto"))
        root.addWidget(self.co_live_backend)

        lbl("Windows音声認識の言語")
        self.co_native_lang = QComboBox(); self.co_native_lang.setStyleSheet(_FIELD)
        for name, value in [("日本語 ja-JP", "ja-JP"), ("英語 en-US", "en-US")]:
            self.co_native_lang.addItem(name, value)
        self._select(self.co_native_lang, cfg.get("native_speech_language", "ja-JP"))
        root.addWidget(self.co_native_lang)

        lbl("ライブ文字起こしモデル（small=推奨・日本語精度優先 / tiny=速度最優先）")
        self.co_live_model = QComboBox(); self.co_live_model.setStyleSheet(_FIELD)
        for m in ["tiny", "base", "small", "medium", "large-v3-turbo"]:
            self.co_live_model.addItem(m, m)
        self._select(self.co_live_model, cfg.get("live_whisper_model", "small"))
        root.addWidget(self.co_live_model)

        self.cb_warm = QCheckBox("⚡ モデルを常駐（連続会議で高速・アイドル解放しない）")
        self.cb_warm.setChecked(cfg.get("keep_warm", False))
        self.cb_warm.setStyleSheet("color:#d0d0ee;")
        root.addWidget(self.cb_warm)

        self.cb_preload = QCheckBox("🚀 起動後に文字起こしモデルを先読み（初回停止後の待ち時間を短縮）")
        self.cb_preload_final = QCheckBox("🎯 最終用高精度モデルも先読み（精度優先・VRAM使用）")
        self.cb_ready_status = QCheckBox("✅ 高精度モデルの準備状態を表示")
        self.cb_fast_summary = QCheckBox("⚡ 停止後10秒モード（軽量な即時要約を使う）")
        self.cb_use_live = QCheckBox("📝 停止時にライブ文字起こし済みの結果を優先（高速だが低精度）")
        self.cb_native_only = QCheckBox("🪟 実験用: Windows純正だけで正式文字起こし（通常はOFF）")
        self.cb_preload.setChecked(cfg.get("preload_transcriber", True))
        self.cb_preload_final.setChecked(cfg.get("preload_final_transcriber", True))
        self.cb_ready_status.setChecked(cfg.get("show_model_ready_status", True))
        self.cb_fast_summary.setChecked(cfg.get("fast_summary", True))
        self.cb_use_live.setChecked(cfg.get("use_live_transcript_on_stop", False))
        self.cb_native_only.setChecked(cfg.get("native_only_transcription", False))
        for c in (self.cb_preload, self.cb_preload_final, self.cb_ready_status, self.cb_fast_summary, self.cb_use_live, self.cb_native_only):
            c.setStyleSheet("color:#d0d0ee;")
            root.addWidget(c)

        lbl("要約モデルのパス（空欄で自動検出 / CTranslate2形式のQwen2.5フォルダ）")
        self.le_summary = QLineEdit(cfg.get("summary_model_dir", "")); self.le_summary.setStyleSheet(_FIELD)
        root.addWidget(self.le_summary)

        lbl("要約バックエンド")
        self.co_backend = QComboBox(); self.co_backend.setStyleSheet(_FIELD)
        self.co_backend.addItem("ローカル CTranslate2", "local")
        self.co_backend.addItem("Ollama", "ollama")
        self.co_backend.addItem("OpenAI互換", "openai_compat")
        self._select(self.co_backend, cfg.get("summary_backend", "local"))
        root.addWidget(self.co_backend)

        lbl("Ollama（URL / model）")
        row_ollama = QHBoxLayout()
        self.le_ollama_url = QLineEdit(cfg.get("ollama_base_url", "http://localhost:11434"))
        self.le_ollama_url.setStyleSheet(_FIELD)
        self.le_ollama_model = QLineEdit(cfg.get("ollama_model", "qwen2.5:3b"))
        self.le_ollama_model.setStyleSheet(_FIELD)
        row_ollama.addWidget(self.le_ollama_url, 1)
        row_ollama.addWidget(self.le_ollama_model, 1)
        root.addLayout(row_ollama)

        lbl("OpenAI互換（base_url / model / api_key）")
        self.le_openai_url = QLineEdit(cfg.get("openai_base_url", "")); self.le_openai_url.setStyleSheet(_FIELD)
        self.le_openai_model = QLineEdit(cfg.get("openai_model", "")); self.le_openai_model.setStyleSheet(_FIELD)
        self.le_openai_key = QLineEdit(cfg.get("openai_api_key", "")); self.le_openai_key.setStyleSheet(_FIELD)
        self.le_openai_key.setEchoMode(QLineEdit.Password)
        root.addWidget(self.le_openai_url)
        root.addWidget(self.le_openai_model)
        root.addWidget(self.le_openai_key)

        lbl("要約セクション（カンマ区切り）")
        sections = cfg.get("summary_sections") or []
        if not isinstance(sections, str):
            sections = ", ".join(str(s) for s in sections)
        self.le_sections = QLineEdit(sections); self.le_sections.setStyleSheet(_FIELD)
        root.addWidget(self.le_sections)

        lbl("要約の追加指示")
        self.te_extra = QTextEdit(cfg.get("summary_extra_instructions", ""))
        self.te_extra.setFixedHeight(72)
        self.te_extra.setStyleSheet(_FIELD)
        root.addWidget(self.te_extra)

        lbl("録音ホットキー（例: ctrl+shift+r）")
        self.le_hotkey = QLineEdit(cfg.get("hotkey", "ctrl+shift+r")); self.le_hotkey.setStyleSheet(_FIELD)
        root.addWidget(self.le_hotkey)

        lbl("保存先フォルダ")
        row = QHBoxLayout()
        self.le_dir = QLineEdit(cfg.get("save_dir", str(RECORDINGS_DIR))); self.le_dir.setStyleSheet(_FIELD)
        browse = QPushButton("…"); browse.setStyleSheet(_FIELD); browse.clicked.connect(self._browse)
        row.addWidget(self.le_dir, 1); row.addWidget(browse)
        root.addLayout(row)

        lbl("予定ICSファイル（空欄ならOutlookのみ / OutlookもOFFなら無効）")
        row_cal = QHBoxLayout()
        self.le_ics = QLineEdit(cfg.get("calendar_ics_path", "")); self.le_ics.setStyleSheet(_FIELD)
        browse_ics = QPushButton("…"); browse_ics.setStyleSheet(_FIELD); browse_ics.clicked.connect(self._browse_ics)
        row_cal.addWidget(self.le_ics, 1); row_cal.addWidget(browse_ics)
        root.addLayout(row_cal)

        self.cb_outlook = QCheckBox("Outlook予定表も参照する（利用可能な場合のみ）")
        self.cb_outlook.setChecked(cfg.get("calendar_outlook_enabled", False))
        self.cb_outlook.setStyleSheet("color:#d0d0ee;")
        root.addWidget(self.cb_outlook)

        root.addStretch(1)
        save_btn = QPushButton("💾  保存して閉じる")
        save_btn.setStyleSheet("QPushButton{background:#3a6fd8;color:#fff;border:none;border-radius:6px;"
                               "padding:10px;font-weight:bold;} QPushButton:hover{background:#2a4fa0;}")
        save_btn.clicked.connect(self._save)
        root.addWidget(save_btn)

    @staticmethod
    def _select(combo, value):
        i = combo.findData(value)
        combo.setCurrentIndex(i if i >= 0 else 0)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "保存先フォルダ", self.le_dir.text())
        if d:
            self.le_dir.setText(d)

    def _browse_ics(self):
        p, _ = QFileDialog.getOpenFileName(self, "予定ICSファイル", self.le_ics.text(), "Calendar (*.ics);;すべて (*.*)")
        if p:
            self.le_ics.setText(p)

    def _save(self):
        if not (self.cb_mic.isChecked() or self.cb_sys.isChecked()):
            QMessageBox.warning(self, "録音対象なし", "マイクまたはシステム音声の少なくとも一方を有効にしてください。")
            return
        self._cfg["record_mic"]        = self.cb_mic.isChecked()
        self._cfg["record_system"]     = self.cb_sys.isChecked()
        self._cfg["enable_aec"]        = self.cb_aec.isChecked()
        self._cfg["enable_diarization"] = self.cb_diar.isChecked()
        self._cfg["enable_live_transcription"] = self.cb_live.isChecked()
        self._cfg["enable_meeting_detection"] = self.cb_detect.isChecked()
        self._cfg["enable_calendar_title_hint"] = self.cb_calendar_title.isChecked()
        self._cfg["mic_name"]          = self.co_mic.currentData()
        self._cfg["speaker_name"]      = self.co_spk.currentData()
        self._cfg["whisper_model"]     = self.co_model.currentData()
        self._cfg["final_channel_policy"] = self.co_final_channel_policy.currentData()
        self._cfg["live_backend"]      = self.co_live_backend.currentData()
        self._cfg["native_speech_language"] = self.co_native_lang.currentData()
        self._cfg["live_whisper_model"] = self.co_live_model.currentData()
        self._cfg["keep_warm"]         = self.cb_warm.isChecked()
        self._cfg["preload_transcriber"] = self.cb_preload.isChecked()
        self._cfg["preload_final_transcriber"] = self.cb_preload_final.isChecked()
        self._cfg["show_model_ready_status"] = self.cb_ready_status.isChecked()
        self._cfg["fast_summary"]      = self.cb_fast_summary.isChecked()
        self._cfg["use_live_transcript_on_stop"] = self.cb_use_live.isChecked()
        self._cfg["native_only_transcription"] = self.cb_native_only.isChecked()
        self._cfg["summary_model_dir"] = self.le_summary.text()
        self._cfg["summary_backend"]   = self.co_backend.currentData()
        self._cfg["ollama_base_url"]   = self.le_ollama_url.text()
        self._cfg["ollama_model"]      = self.le_ollama_model.text()
        self._cfg["openai_base_url"]   = self.le_openai_url.text()
        self._cfg["openai_model"]      = self.le_openai_model.text()
        self._cfg["openai_api_key"]    = self.le_openai_key.text()
        self._cfg["summary_sections"]  = [s.strip() for s in self.le_sections.text().split(",") if s.strip()]
        self._cfg["summary_extra_instructions"] = self.te_extra.toPlainText()
        self._cfg["hotkey"]            = self.le_hotkey.text()
        self._cfg["save_dir"]          = self.le_dir.text()
        self._cfg["calendar_ics_path"] = self.le_ics.text()
        self._cfg["calendar_outlook_enabled"] = self.cb_outlook.isChecked()
        save_config(self._cfg)
        self._on_save(self._cfg)
        self.accept()
