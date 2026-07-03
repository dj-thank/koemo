"""Koemo — エントリポイント（`pythonw koemo.pyw` で起動）。"""
import os
import sys
import ctypes

# Windows DPI 高解像度対応
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from koemo.gpu import enable_cuda_dlls
enable_cuda_dlls()

from koemo.app import main

if __name__ == "__main__":
    main()
