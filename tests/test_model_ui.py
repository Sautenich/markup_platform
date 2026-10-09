from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLabel

from lerobot_studio.config import DatasetRegistry
from lerobot_studio.ui.home_page import HomePage
from lerobot_studio.ui.model_pages import DatasetChecklist, TrainingPage


def _registry(tmp_path, count=3):
    paths = []
    for index in range(count):
        path = tmp_path / f"dataset_{index}"
        path.mkdir()
        paths.append(path)
    registry = DatasetRegistry(tmp_path / "datasets.json")
    registry.save(paths)
    return registry, paths


def test_training_dataset_can_be_selected_by_clicking_anywhere_on_row(qtbot, tmp_path):
    registry, paths = _registry(tmp_path)
    checklist = DatasetChecklist(registry, multiple=True)
    qtbot.addWidget(checklist)
    checklist.resize(700, 300)
    checklist.show()

    QTest.mouseClick(checklist.viewport(), Qt.LeftButton, pos=checklist.visualItemRect(checklist.item(0)).center())
    QTest.mouseClick(checklist.viewport(), Qt.LeftButton, pos=checklist.visualItemRect(checklist.item(1)).center())

    assert checklist.selected_paths() == paths[:2]


def test_testing_dataset_selection_is_single(qtbot, tmp_path):
    registry, paths = _registry(tmp_path)
    checklist = DatasetChecklist(registry, multiple=False)
    qtbot.addWidget(checklist)
    checklist.resize(700, 300)
    checklist.show()

    QTest.mouseClick(checklist.viewport(), Qt.LeftButton, pos=checklist.visualItemRect(checklist.item(0)).center())
    QTest.mouseClick(checklist.viewport(), Qt.LeftButton, pos=checklist.visualItemRect(checklist.item(1)).center())

    assert checklist.selected_paths() == [paths[1]]


def test_dry_run_pages_do_not_show_top_banner(qtbot, tmp_path):
    registry, _paths = _registry(tmp_path, count=1)
    home = HomePage(dry_run=True)
    training = TrainingPage(tmp_path, registry, dry_run=True)
    qtbot.addWidget(home)
    qtbot.addWidget(training)

    assert not any("DRY RUN" in label.text() for label in home.findChildren(QLabel))
    assert not any("DRY RUN" in label.text() for label in training.findChildren(QLabel))
