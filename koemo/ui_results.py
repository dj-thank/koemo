"""結果ウィンドウ（要約＋文字起こし）。Markdownを整形表示。"""
import os
import re
from pathlib import Path

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QTabWidget, QTextBrowser, QPushButton, QApplication, QFileDialog)
from PySide6.QtCore import Qt

from .export import export_pdf, export_docx, export_markdown
from .ui_chat import ChatWindow

_BTN = ("QPushButton{background:#1e1e3a;color:#fff;border:none;border-radius:6px;"
        "padding:8px 14px;} QPushButton:hover{background:#2a2a4a;}")
_BTN_ACCENT = ("QPushButton{background:#3a6fd8;color:#fff;border:none;border-radius:6px;"
               "padding:8px 14px;} QPushButton:hover{background:#2a4fa0;}")


class ResultsWindow(QWidget):
    def __init__(self, title, summary_md, transcript_md, save_dir, duration, chat_func=None):
        super().__init__()
        self._summary_md = summary_md
        self._transcript_md = transcript_md
        self._save_dir = Path(save_dir)
        self._title = title
        self._doc = f"# {title}\n\n{summary_md}\n\n---\n\n## 文字起こし\n\n{transcript_md}\n"
        self._chat_func = chat_func
        self._chat = None

        self.setWindowTitle(f"Koemo — {title}")
        self.resize(800, 700)
        self.setStyleSheet("background:#0d0d1a;")

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 18)
        root.setSpacing(12)

        head = QHBoxLayout()
        t = QLabel(f"🎙  {title}")
        t.setStyleSheet("color:#a0c4ff;font-size:18px;font-weight:bold;")
        t.setWordWrap(True)
        m, s = divmod(int(duration), 60)
        dur = QLabel(f"{m:02d}:{s:02d}")
        dur.setStyleSheet("color:#667;font-size:12px;")
        head.addWidget(t, 1)
        head.addWidget(dur, 0, Qt.AlignTop)
        root.addLayout(head)

        tabs = QTabWidget()
        tabs.setStyleSheet(
            "QTabWidget::pane{border:none;background:#111126;}"
            "QTabBar::tab{background:#1a1a2e;color:#8888aa;padding:8px 16px;border:none;}"
            "QTabBar::tab:selected{background:#252540;color:#a0c4ff;}")
        self._sum = self._make_view(summary_md)
        self._tr = self._make_view(transcript_md)
        tabs.addTab(self._sum, "📋  要約")
        tabs.addTab(self._tr, "📝  文字起こし")
        root.addWidget(tabs, 1)

        bar = QHBoxLayout()
        b1 = QPushButton("📋 要約をコピー"); b1.setStyleSheet(_BTN_ACCENT); b1.clicked.connect(self._copy_sum)
        b2 = QPushButton("📝 文字起こしをコピー"); b2.setStyleSheet(_BTN); b2.clicked.connect(self._copy_tr)
        b3 = QPushButton("📁 フォルダを開く"); b3.setStyleSheet(_BTN); b3.clicked.connect(self._open)
        b4 = QPushButton("📄 PDF"); b4.setStyleSheet(_BTN); b4.clicked.connect(self._export_pdf)
        b5 = QPushButton("📑 DOCX"); b5.setStyleSheet(_BTN); b5.clicked.connect(self._export_docx)
        b6 = QPushButton("MD"); b6.setStyleSheet(_BTN); b6.clicked.connect(self._export_markdown)
        b7 = QPushButton("💬 チャット"); b7.setStyleSheet(_BTN); b7.clicked.connect(self._open_chat)
        bar.addWidget(b1); bar.addWidget(b2); bar.addWidget(b3)
        bar.addWidget(b4); bar.addWidget(b5); bar.addWidget(b6); bar.addStretch(1)
        if self._chat_func:
            bar.insertWidget(2, b7)
        root.addLayout(bar)

    def _make_view(self, md):
        v = QTextBrowser()
        v.setOpenExternalLinks(True)
        v.setStyleSheet("background:#111126;color:#d0d0ee;border:none;font-size:14px;padding:8px;")
        v.setMarkdown(md)
        return v

    def _copy_sum(self):
        QApplication.clipboard().setText(self._summary_md)

    def _copy_tr(self):
        QApplication.clipboard().setText(self._transcript_md)

    def _open(self):
        try:
            os.startfile(str(self._save_dir))
        except Exception:
            pass

    def _open_chat(self):
        if not self._chat_func:
            return
        self._chat = ChatWindow(self._title, self._chat_func)
        self._chat.show()
        self._chat.raise_()
        self._chat.activateWindow()

    def _safe_name(self):
        return re.sub(r'[\\/:*?"<>|]', "_", self._title)[:60] or "koemo"

    def _export_pdf(self):
        p, _ = QFileDialog.getSaveFileName(self, "PDFで保存",
                                           str(self._save_dir / f"{self._safe_name()}.pdf"), "PDF (*.pdf)")
        if p:
            export_pdf(self._doc, p)

    def _export_docx(self):
        p, _ = QFileDialog.getSaveFileName(self, "Wordで保存",
                                           str(self._save_dir / f"{self._safe_name()}.docx"), "Word (*.docx)")
        if p:
            export_docx(self._doc, p)

    def _export_markdown(self):
        p, _ = QFileDialog.getSaveFileName(self, "Markdownで保存",
                                           str(self._save_dir / f"{self._safe_name()}.md"), "Markdown (*.md)")
        if p:
            export_markdown(self._doc, p)
