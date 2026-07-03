"""Koemo ログ — `~/.koemo/logs/koemo.log` へ回転記録し、未捕捉例外も残す。

`pythonw` 起動では stderr が捨てられるため、クラッシュ原因はここにしか残らない。
"""
import logging
import logging.handlers
import sys
import threading

from .config import CONFIG_DIR

LOG_DIR = CONFIG_DIR / "logs"
LOG_FILE = LOG_DIR / "koemo.log"


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    if any(isinstance(h, logging.handlers.RotatingFileHandler) for h in root.handlers):
        return LOG_FILE
    handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    def _hook(exc_type, exc, tb):
        logging.getLogger("koemo").critical(
            "uncaught exception", exc_info=(exc_type, exc, tb))
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _hook

    def _thread_hook(args):
        name = args.thread.name if args.thread else "?"
        logging.getLogger("koemo").critical(
            "uncaught thread exception in %s", name,
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    threading.excepthook = _thread_hook
    return LOG_FILE
