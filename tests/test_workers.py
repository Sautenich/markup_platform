from __future__ import annotations

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication, QWidget

from lerobot_studio.ui.dialogs import AugmentationDialog
from lerobot_studio.ui.workers import run_augmentation
from test_augment import make_dataset


def test_augmentation_worker_keeps_qt_event_loop_alive(qtbot, tmp_path, monkeypatch):
    dataset = make_dataset(tmp_path / "dataset", episodes=1)
    parent = QWidget()
    qtbot.addWidget(parent)
    update_threads = []

    from lerobot_studio.ui.dialogs import OperationProgress

    original_set_value = OperationProgress.setValue

    def record_update_thread(dialog, value):
        update_threads.append(QThread.currentThread())
        original_set_value(dialog, value)

    monkeypatch.setattr(OperationProgress, "setValue", record_update_thread)
    created = run_augmentation(parent, dataset, "signals", "joint_noise", 1, "Test")
    assert len(created) == 1
    assert any(item.index == created[0] for item in dataset.episodes(refresh=True))
    assert update_threads
    assert all(thread is QApplication.instance().thread() for thread in update_threads)


def test_visual_slider_is_always_editable_and_enables_effect(qtbot):
    dialog = AugmentationDialog(5)
    qtbot.addWidget(dialog)
    dialog.tabs.setCurrentIndex(1)
    row = dialog.visual_rows["brightness"]
    assert row.slider.isEnabled()
    assert row.spin.isEnabled()
    assert not row.enabled.isChecked()
    row.slider.setValue(row.slider.value() + 1)
    assert row.enabled.isChecked()
    mode, parameters, count = dialog.request()
    assert mode == "visual"
    assert "brightness" in parameters
    assert count == 5
