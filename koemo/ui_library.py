"""会議履歴ウィンドウ。"""
from pathlib import Path

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QListWidget, QListWidgetItem)
from PySide6.QtCore import Qt

from . import library
from .ui_results import ResultsWindow

_FIELD = "background:#1a1a2e;color:#d0d0ee;border:none;border-radius:6px;padding:8px;"
_BTN = ("QPushButton{background:#3a6fd8;color:#fff;border:none;border-radius:6px;"
        "padding:8px 14px;} QPushButton:hover{background:#2a4fa0;}")


class LibraryWindow(QWidget):
    def __init__(self, chat_factory=None):
        super().__init__()
        self._chat_factory = chat_factory
        self._results = []
        self.setWindowTitle("Koemo — 履歴")
        self.resize(760, 620)
        self.setStyleSheet("background:#0d0d1a;color:#d0d0ee;")

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 18)
        root.setSpacing(10)

        title = QLabel("会議履歴")
        title.setStyleSheet("color:#a0c4ff;font-size:18px;font-weight:bold;")
        root.addWidget(title)

        row = QHBoxLayout()
        self._query = QLineEdit()
        self._query.setStyleSheet(_FIELD)
        self._query.returnPressed.connect(self._reload)
        b = QPushButton("検索")
        b.setStyleSheet(_BTN)
        b.clicked.connect(self._reload)
        row.addWidget(self._query, 1)
        row.addWidget(b)
        root.addLayout(row)

        self._list = QListWidget()
        self._list.setStyleSheet(
            "QListWidget{background:#111126;color:#d0d0ee;border:none;border-radius:6px;padding:6px;}"
            "QListWidget::item{padding:10px;border-bottom:1px solid #252540;}"
            "QListWidget::item:selected{background:#252540;color:#ffffff;}")
        self._list.itemActivated.connect(self._open_item)
        self._list.itemDoubleClicked.connect(self._open_item)
        root.addWidget(self._list, 1)

        self._reload()

    def _reload(self):
        self._list.clear()
        for row in library.search(self._query.text()):
            dur = int(row.get("duration") or 0)
            m, s = divmod(dur, 60)
            item = QListWidgetItem(f"{row.get('date') or ''}  {row.get('title') or '会議メモ'}  ({m:02d}:{s:02d})")
            item.setData(Qt.UserRole, row["id"])
            self._list.addItem(item)

    def _open_item(self, item):
        row = library.get(item.data(Qt.UserRole))
        if not row:
            return
        chat_func = None
        if self._chat_factory:
            chat_func = self._chat_factory(row.get("transcript_md") or "")
        win = ResultsWindow(
            row.get("title") or "会議メモ",
            row.get("summary_md") or "",
            row.get("transcript_md") or "",
            Path(row.get("save_dir") or "."),
            row.get("duration") or 0,
            chat_func=chat_func,
        )
        self._results.append(win)
        win.show()
        win.raise_()
        win.activateWindow()
