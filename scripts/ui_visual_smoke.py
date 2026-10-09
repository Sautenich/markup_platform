#!/usr/bin/env python3
"""Drive one real visual augmentation through the Qt widgets.

This is intentionally not a service-level shortcut: it clicks the episode page
button, selects the visual tab, changes a slider with a keyboard UI event, sets
the requested episode count, and accepts the same dialogs a user sees.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PySide6.QtCore import QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QMessageBox, QPushButton

from lerobot_studio.debug import configure_logging, install_qt_message_logging
from lerobot_studio.services.lerobot import LeRobotDataset
from lerobot_studio.ui.dialogs import AugmentationDialog
from lerobot_studio.ui.episode_page import EpisodePage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--mode", choices=("visual", "signals"), default="visual")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--timeout-ms", type=int, default=180_000)
    args = parser.parse_args()
    configure_logging()
    install_qt_message_logging()

    app = QApplication(sys.argv[:1])
    page = EpisodePage()
    page.resize(1600, 1000)
    dataset = LeRobotDataset(args.dataset)
    before = {episode.index for episode in dataset.episodes(refresh=True)}
    page.open_dataset(dataset)
    page.show()

    state = {"dialog_configured": False, "done": False, "error": None, "slider": None}

    def fail(message: str) -> None:
        state["error"] = message
        state["done"] = True
        app.exit(2)

    def watch_modals() -> None:
        modal = app.activeModalWidget()
        if isinstance(modal, AugmentationDialog) and not state["dialog_configured"]:
            state["dialog_configured"] = True
            if args.mode == "visual":
                tab_bar = modal.tabs.tabBar()
                QTest.mouseClick(tab_bar, Qt.LeftButton, pos=tab_bar.tabRect(1).center())
                row = modal.visual_rows["brightness"]
                old_value = row.slider.value()
                row.slider.setFocus()
                QTest.keyClick(row.slider, Qt.Key_Right)
                new_value = row.slider.value()
                state["slider"] = {"before": old_value, "after": new_value, "checked": row.enabled.isChecked()}
                if new_value == old_value or not row.enabled.isChecked():
                    fail("Ползунок яркости не изменился или эффект не включился")
                    return
            modal.count.setFocus()
            QTest.keyClick(modal.count, Qt.Key_A, Qt.ControlModifier)
            QTest.keyClicks(modal.count, str(args.count))
            buttons = modal.findChild(QDialogButtonBox)
            QTest.mouseClick(buttons.button(QDialogButtonBox.Ok), Qt.LeftButton)
            return
        if isinstance(modal, QMessageBox):
            title = modal.windowTitle()
            text = modal.text()
            if "Ошибка" in title:
                fail(f"{title}: {text}")
                return
            if title == "Готово":
                QTest.mouseClick(modal.button(QMessageBox.Ok), Qt.LeftButton)
                state["done"] = True
                QTimer.singleShot(0, app.quit)

    watcher = QTimer()
    watcher.timeout.connect(watch_modals)
    watcher.start(100)

    def click_augmentation() -> None:
        button = next(
            (item for item in page.findChildren(QPushButton) if "Аугментации" in item.text()),
            None,
        )
        if button is None:
            fail("Кнопка аугментации не найдена")
            return
        QTest.mouseClick(button, Qt.LeftButton)

    QTimer.singleShot(200, click_augmentation)
    QTimer.singleShot(args.timeout_ms, lambda: fail("UI smoke test timeout") if not state["done"] else None)
    exit_code = app.exec()
    page.stop()

    after_dataset = LeRobotDataset(args.dataset)
    after = {episode.index for episode in after_dataset.episodes(refresh=True)}
    created = sorted(after - before)
    result = {
        "ok": exit_code == 0 and state["error"] is None and len(created) == args.count,
        "created": created,
        "slider": state["slider"],
        "error": state["error"],
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
