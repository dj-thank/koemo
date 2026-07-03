"""会議内容へのQ&Aチャット。"""
import html
import threading

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTextBrowser,
                               QLineEdit, QPushButton, QLabel)
from PySide6.QtCore import Signal, Slot
from PySide6.QtGui import QFont

_BTN = ("QPushButton{background:#3a6fd8;color:#fff;border:none;border-radius:6px;"
        "padding:8px 14px;font-weight:bold;} QPushButton:hover{background:#2a4fa0;}"
        "QPushButton:disabled{background:#303048;color:#8888aa;}")
_FIELD = "background:#1a1a2e;color:#d0d0ee;border:none;border-radius:6px;padding:8px;"


class ChatWindow(QWidget):
    answered = Signal(str)
    failed = Signal(str)

    def __init__(self, title, ask_func):
        super().__init__()
        self._ask_func = ask_func
        self._history = []
        self._busy = False
        self.setWindowTitle(f"Koemo — チャット — {title}")
        self.resize(680, 560)
        self.setStyleSheet("background:#0d0d1a;color:#d0d0ee;")

        root = QVBoxLayout(self)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(10)

        head = QLabel("会議チャット")
        head.setStyleSheet("color:#a0c4ff;font-size:18px;font-weight:bold;")
        head.setFont(QFont("Yu Gothic UI", 12))
        root.addWidget(head)

        self._view = QTextBrowser()
        self._view.setStyleSheet(
            "background:#111126;color:#d0d0ee;border:none;border-radius:6px;"
            "font-size:14px;padding:10px;")
        root.addWidget(self._view, 1)

        row = QHBoxLayout()
        self._input = QLineEdit()
        self._input.setStyleSheet(_FIELD)
        self._input.returnPressed.connect(self._ask)
        self._send = QPushButton("送信")
        self._send.setStyleSheet(_BTN)
        self._send.clicked.connect(self._ask)
        row.addWidget(self._input, 1)
        row.addWidget(self._send)
        root.addLayout(row)

        self.answered.connect(self._on_answered)
        self.failed.connect(self._on_failed)

    @Slot()
    def _ask(self):
        if self._busy:
            return
        question = self._input.text().strip()
        if not question:
            return
        self._input.clear()
        self._append("あなた", question)
        history = list(self._history)
        self._history.append(("user", question))
        self._busy = True
        self._send.setEnabled(False)
        threading.Thread(target=self._worker, args=(question, history), daemon=True).start()

    def _worker(self, question, history):
        try:
            self.answered.emit(self._ask_func(question, history))
        except Exception as e:
            self.failed.emit(str(e))

    @Slot(str)
    def _on_answered(self, answer):
        answer = answer or "回答を生成できませんでした。"
        self._history.append(("assistant", answer))
        self._append("Koemo", answer)
        self._busy = False
        self._send.setEnabled(True)

    @Slot(str)
    def _on_failed(self, msg):
        self._append("エラー", msg)
        self._busy = False
        self._send.setEnabled(True)

    def _append(self, who, text):
        self._view.append(f"<b>{html.escape(who)}</b>: {html.escape(text)}")
