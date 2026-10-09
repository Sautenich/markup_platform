from __future__ import annotations

import os
import signal
import socket
import sys
import logging
import argparse
import time
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QLockFile, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from lerobot_studio.config import config_dir
from lerobot_studio.debug import configure_logging, install_qt_message_logging
from lerobot_studio.ui.main_window import MainWindow


STYLE = """
QWidget { background: #0d1117; color: #dce5ef; font-size: 13px; }
QMainWindow, QStackedWidget { background: #0d1117; }
QLabel#pageTitle { font-size: 30px; font-weight: 700; color: #f0f6fc; }
QLabel#sectionTitle { font-size: 19px; font-weight: 700; color: #f0f6fc; }
QLabel#subtitle { color: #8b9aaa; font-size: 14px; margin-bottom: 15px; }
QLabel#promptLabel { background: #151c25; border: 1px solid #283341; border-radius: 7px; padding: 9px; color: #b9c8d8; }
QPushButton { background: #1b2530; border: 1px solid #344252; border-radius: 6px; padding: 7px 11px; }
QPushButton:hover { background: #253241; border-color: #52667c; }
QPushButton:pressed { background: #111820; }
QPushButton:disabled { color: #5e6a77; background: #141a21; }
QPushButton#primaryButton { background: #2367d1; border-color: #3380ed; color: white; font-weight: 600; }
QPushButton#dangerButton { color: #ff9b98; border-color: #6b3537; }
QPushButton#playButton { font-size: 18px; min-width: 32px; }
QTableWidget { background: #111820; alternate-background-color: #151e28; border: 1px solid #293442; border-radius: 7px; gridline-color: #202a35; }
QTableWidget::item { padding: 8px; }
QTableWidget::item:selected { background: #1f579c; }
QHeaderView::section { background: #18212b; color: #9eb0c3; padding: 8px; border: 0; border-bottom: 1px solid #344252; }
QComboBox, QSpinBox, QPlainTextEdit { background: #111820; border: 1px solid #344252; border-radius: 5px; padding: 5px; selection-background-color: #2467ad; }
QLineEdit, QDoubleSpinBox { background: #111820; border: 1px solid #344252; border-radius: 5px; padding: 6px; selection-background-color: #2467ad; }
QListWidget { background: #111820; border: 1px solid #293442; border-radius: 7px; padding: 4px; }
QListWidget::item { padding: 8px; border-bottom: 1px solid #202a35; }
QSlider::groove:horizontal { height: 4px; background: #2a3745; border-radius: 2px; }
QSlider::handle:horizontal { background: #58a6ff; width: 14px; margin: -5px 0; border-radius: 7px; }
QFrame#tipCard, QFrame#helpCard { background: #121b24; border: 1px solid #283746; border-radius: 8px; }
QFrame#cameraPane { background: black; border: 1px solid #2a3745; border-radius: 6px; }
QLabel#cameraTitle { background: #111820; color: #99aabd; padding: 4px 8px; }
QLabel#helpTitle { color: #f0f6fc; font-size: 15px; font-weight: 600; }
QLabel#keyLabel { background: #273443; border: 1px solid #405166; border-radius: 4px; padding: 2px 5px; color: white; }
QTabWidget::pane { border: 1px solid #293442; }
QTabBar::tab { background: #151e28; padding: 8px 14px; }
QTabBar::tab:selected { background: #245b92; }
QFrame#menuCard, QFrame#settingsCard { background: #121a23; border: 1px solid #2b3948; border-radius: 12px; }
QFrame#menuCard:hover { border-color: #4d6985; }
QLabel#menuNumber { background: transparent; color: #58a6ff; font-size: 15px; font-weight: 700; }
QLabel#menuTitle { background: transparent; color: #f0f6fc; font-size: 21px; font-weight: 700; }
QLabel#menuDescription, QLabel#mutedLabel { background: transparent; color: #8b9aaa; }
QLabel#statusBadge { background: #183b2b; color: #83e6ae; border-radius: 5px; padding: 5px 10px; }
QPlainTextEdit#terminal { background: #070a0e; color: #b7f5ca; border: 1px solid #293442; border-radius: 7px; padding: 10px; }
QFrame#metricsBar { background: #121a23; border: 1px solid #2b3948; border-radius: 8px; }
QLabel#metricName { color: #8493a5; font-size: 11px; }
QLabel#metricValue { color: #f0f6fc; font-size: 18px; font-weight: 700; min-width: 110px; }
"""


def _is_studio_process(pid: int) -> bool:
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            errors="replace"
        )
    except OSError:
        return False
    markers = ("lerobot-studio", "lerobot_studio.app", "lerobot_studio/app.py")
    return any(marker in command for marker in markers)


def acquire_instance_lock(
    path: Path,
    *,
    replace_existing: bool = False,
    shutdown_timeout: float = 5.0,
) -> QLockFile | None:
    """Acquire the lock, optionally closing and replacing the existing Studio."""
    instance_lock = QLockFile(str(path))
    instance_lock.setStaleLockTime(30_000)
    if instance_lock.tryLock(100):
        return instance_lock
    if not replace_existing:
        return None
    try:
        owner_pid, owner_host, _owner_app = instance_lock.getLockInfo()
    except (RuntimeError, TypeError, ValueError):
        return None
    if (
        owner_pid <= 0
        or owner_pid == os.getpid()
        or owner_host != socket.gethostname()
        or not _is_studio_process(owner_pid)
    ):
        logging.getLogger(__name__).error(
            "Refusing to terminate unverified lock owner pid=%s host=%s", owner_pid, owner_host
        )
        return None
    logging.getLogger(__name__).info("Closing previous LeRobot Studio pid=%s", owner_pid)
    try:
        os.kill(owner_pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError:
        logging.getLogger(__name__).exception("Cannot terminate previous LeRobot Studio")
        return None

    deadline = time.monotonic() + shutdown_timeout
    while time.monotonic() < deadline:
        if instance_lock.tryLock(100):
            return instance_lock
        try:
            os.kill(owner_pid, 0)
        except ProcessLookupError:
            instance_lock.removeStaleLockFile()
        time.sleep(0.05)

    # A stuck modal/native operation may not process the graceful Qt shutdown.
    # The target was verified from this application's private lock and cmdline.
    if _is_studio_process(owner_pid):
        logging.getLogger(__name__).warning(
            "Previous LeRobot Studio did not close gracefully; killing pid=%s", owner_pid
        )
        try:
            os.kill(owner_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if instance_lock.tryLock(100):
                return instance_lock
            instance_lock.removeStaleLockFile()
            time.sleep(0.05)
    return None


def main(argv: list[str] | None = None) -> int:
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="LeRobot Studio")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="симулировать обучение и тестирование без запуска моделей",
    )
    arguments, qt_arguments = parser.parse_known_args(raw_arguments)
    log_path = configure_logging()
    install_qt_message_logging()
    logging.getLogger(__name__).info("Starting LeRobot Studio pid=%s", os.getpid())
    # A GUI launched from a shell must not be suspended by terminal job-control
    # signals when Qt/FFmpeg writes a diagnostic message.
    for signal_name in ("SIGTTOU", "SIGTTIN", "SIGTSTP"):
        terminal_signal = getattr(signal, signal_name, None)
        if terminal_signal is not None:
            signal.signal(terminal_signal, signal.SIG_IGN)
    current_rules = os.environ.get("QT_LOGGING_RULES", "")
    quiet_multimedia = "qt.multimedia.ffmpeg=false;qt.multimedia.playbackengine=false"
    os.environ["QT_LOGGING_RULES"] = f"{current_rules};{quiet_multimedia}" if current_rules else quiet_multimedia
    QCoreApplication.setOrganizationName("Skoltech")
    QCoreApplication.setApplicationName("LeRobot Studio")
    app = QApplication([sys.argv[0], *qt_arguments])
    app.setStyle("Fusion")
    app.setFont(QFont("Inter", 10))
    app.setStyleSheet(STYLE)
    instance_lock = acquire_instance_lock(
        config_dir() / "instance.lock",
        replace_existing=True,
    )
    if instance_lock is None:
        logging.getLogger(__name__).error(
            "Could not close the previous LeRobot Studio instance; aborting replacement launch"
        )
        return 1
    app.instance_lock = instance_lock
    workspace = Path(os.getenv("LEROBOT_STUDIO_WORKSPACE", Path.cwd())).resolve()
    logging.getLogger(__name__).info("Workspace=%s debug_log=%s", workspace, log_path)
    window = MainWindow(workspace, dry_run=arguments.dry_run)
    window.show()

    def close_for_replacement(_signum=None, _frame=None) -> None:
        logging.getLogger(__name__).info("Replacement launch requested; closing this instance")
        QTimer.singleShot(0, window.close)
        QTimer.singleShot(0, app.quit)

    signal.signal(signal.SIGTERM, close_for_replacement)
    # Periodically return control to Python so SIGTERM is handled even while Qt
    # is otherwise idle inside its native event loop.
    signal_timer = QTimer(app)
    signal_timer.timeout.connect(lambda: None)
    signal_timer.start(200)
    app.signal_timer = signal_timer
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
