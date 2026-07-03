"""ライブ字幕ウィンドウ。"""
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextBrowser, QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QTextCursor


class LiveWindow(QWidget):
    def __init__(self):
        super().__init__(None, Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setWindowTitle("Koemo — Live")
        self.resize(520, 220)
        self.setStyleSheet("background:#0d0d1a;color:#d0d0ee;")

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(8)

        title = QLabel("ライブ文字起こし")
        title.setStyleSheet("color:#a0c4ff;font-size:14px;font-weight:bold;")
        title.setFont(QFont("Yu Gothic UI", 10))
        root.addWidget(title)

        self._view = QTextBrowser()
        self._view.setStyleSheet(
            "background:#111126;color:#f0f0ff;border:1px solid #252540;"
            "border-radius:6px;font-size:16px;padding:10px;")
        self._view.setFont(QFont("Yu Gothic UI", 12))
        self._view.setPlainText("")
        root.addWidget(self._view, 1)

    def show_at_corner(self):
        g = QApplication.primaryScreen().availableGeometry()
        self.move(g.right() - self.width() - 18, g.bottom() - self.height() - 128)
        self.show()
        self.raise_()

    def update_text(self, text):
        self._view.setPlainText(text)
        self._view.moveCursor(QTextCursor.MoveOperation.End)
