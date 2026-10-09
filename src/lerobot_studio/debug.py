from __future__ import annotations

import logging
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

from lerobot_studio.config import config_dir


LOG_PATH = config_dir() / "lerobot-studio.log"
_QT_MESSAGE_HANDLER = None


def configure_logging() -> Path:
    """Configure a persistent log early enough to catch GUI/worker failures."""
    root = logging.getLogger()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s"
    )
    if not any(isinstance(handler, RotatingFileHandler) for handler in root.handlers):
        handler = RotatingFileHandler(
            LOG_PATH,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)
    if not any(getattr(handler, "name", None) == "lerobot-studio-console" for handler in root.handlers):
        console = logging.StreamHandler(sys.stderr)
        console.set_name("lerobot-studio-console")
        console.setFormatter(formatter)
        root.addHandler(console)
    root.setLevel(logging.INFO)

    previous_hook = sys.excepthook

    def exception_hook(exc_type, exc_value, traceback) -> None:
        logging.getLogger("lerobot_studio.crash").critical(
            "Unhandled Python exception", exc_info=(exc_type, exc_value, traceback)
        )
        previous_hook(exc_type, exc_value, traceback)

    sys.excepthook = exception_hook

    def thread_exception_hook(args: threading.ExceptHookArgs) -> None:
        logging.getLogger("lerobot_studio.crash").critical(
            "Unhandled thread exception in %s",
            args.thread.name if args.thread else "unknown",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    threading.excepthook = thread_exception_hook
    logging.getLogger(__name__).info("Logging initialized: %s", LOG_PATH)
    return LOG_PATH


def install_qt_message_logging() -> None:
    """Send Qt/FFmpeg/QBackingStore diagnostics to the persistent app log."""
    from PySide6.QtCore import QtMsgType, qInstallMessageHandler

    levels = {
        QtMsgType.QtDebugMsg: logging.DEBUG,
        QtMsgType.QtInfoMsg: logging.INFO,
        QtMsgType.QtWarningMsg: logging.WARNING,
        QtMsgType.QtCriticalMsg: logging.ERROR,
        QtMsgType.QtFatalMsg: logging.CRITICAL,
    }

    def handler(message_type, context, message: str) -> None:
        category = getattr(context, "category", None) or "qt"
        location = ""
        if getattr(context, "file", None):
            location = f" ({context.file}:{context.line})"
        logging.getLogger(f"qt.{category}").log(
            levels.get(message_type, logging.WARNING), "%s%s", message, location
        )

    global _QT_MESSAGE_HANDLER
    _QT_MESSAGE_HANDLER = handler
    qInstallMessageHandler(_QT_MESSAGE_HANDLER)
