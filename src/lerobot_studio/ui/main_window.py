from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QProcess, Qt
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox, QProgressDialog, QStackedWidget

from lerobot_studio.config import DatasetRegistry
from lerobot_studio.services.export import SplitExporter
from lerobot_studio.services.lerobot import LeRobotDataset
from lerobot_studio.ui.datasets_page import DatasetsPage
from lerobot_studio.ui.dialogs import AugmentationDialog, SplitDialog
from lerobot_studio.ui.episode_page import EpisodePage
from lerobot_studio.ui.home_page import HomePage
from lerobot_studio.ui.job_window import JobWindow
from lerobot_studio.ui.model_pages import TestingPage, TrainingPage
from lerobot_studio.ui.workers import run_augmentation


class MainWindow(QMainWindow):
    def __init__(self, workspace: Path, dry_run: bool = False) -> None:
        super().__init__()
        self.workspace = workspace
        self.dry_run = dry_run
        self.job_windows: list[JobWindow] = []
        self.registry = DatasetRegistry()
        self._discover_local_datasets()
        self.setWindowTitle("LeRobot Studio")
        self.resize(1480, 900)
        self.stack = QStackedWidget()
        self.home_page = HomePage(dry_run)
        self.datasets_page = DatasetsPage(self.registry)
        self.training_page = TrainingPage(workspace, self.registry, dry_run)
        self.testing_page = TestingPage(workspace, self.registry, dry_run)
        self.episode_page = EpisodePage()
        self.stack.addWidget(self.home_page)
        self.stack.addWidget(self.datasets_page)
        self.stack.addWidget(self.training_page)
        self.stack.addWidget(self.testing_page)
        self.stack.addWidget(self.episode_page)
        self.setCentralWidget(self.stack)
        self.home_page.datasets_requested.connect(self.show_datasets)
        self.home_page.training_requested.connect(self.show_training)
        self.home_page.testing_requested.connect(self.show_testing)
        self.datasets_page.home_requested.connect(self.show_home)
        self.datasets_page.open_requested.connect(self.open_dataset)
        self.datasets_page.split_requested.connect(self.create_splits)
        self.datasets_page.settings_requested.connect(self.dataset_settings)
        self.episode_page.back_requested.connect(self.show_datasets)
        self.episode_page.dataset_changed.connect(self.datasets_page.refresh)
        self.training_page.home_requested.connect(self.show_home)
        self.testing_page.home_requested.connect(self.show_home)
        self.training_page.job_requested.connect(self.start_job)
        self.testing_page.job_requested.connect(self.start_job)

    def _discover_local_datasets(self) -> None:
        data_root = self.workspace / "datasets"
        if not data_root.exists():
            return
        for child in data_root.iterdir():
            if child.is_dir() and LeRobotDataset.validate(child):
                self.registry.add(child)

    def open_dataset(self, path: Path) -> None:
        try:
            dataset = LeRobotDataset(path)
            self.episode_page.open_dataset(dataset)
        except Exception as error:
            QMessageBox.critical(self, "Не удалось открыть датасет", str(error))
            return
        self.stack.setCurrentWidget(self.episode_page)

    def show_datasets(self) -> None:
        self.episode_page.stop()
        self.datasets_page.refresh()
        self.stack.setCurrentWidget(self.datasets_page)

    def show_home(self) -> None:
        self.episode_page.stop()
        self.stack.setCurrentWidget(self.home_page)

    def show_training(self) -> None:
        self.training_page.datasets.refresh()
        self.stack.setCurrentWidget(self.training_page)

    def show_testing(self) -> None:
        self.testing_page.datasets.refresh()
        self.testing_page.refresh_checkpoints()
        self.stack.setCurrentWidget(self.testing_page)

    def start_job(self, spec) -> None:
        window = JobWindow(spec, self)
        window.setAttribute(Qt.WA_DeleteOnClose, True)
        window.destroyed.connect(lambda: self._forget_job(window))
        window.completed.connect(lambda _spec: self.testing_page.refresh_checkpoints())
        self.job_windows.append(window)
        window.show()

    def _forget_job(self, window: JobWindow) -> None:
        if window in self.job_windows:
            self.job_windows.remove(window)

    def create_splits(self, path: Path) -> None:
        dialog = SplitDialog(self)
        if not dialog.exec():
            return
        selected = QFileDialog.getExistingDirectory(self, "Папка для новых split-датасетов")
        if not selected:
            return
        dataset = LeRobotDataset(path)
        partitions = SplitExporter(dataset).partition(*dialog.values())
        total = sum(len(items) for items in partitions.values())
        progress = QProgressDialog("Подготовка split-датасетов…", "", 0, total, self)
        progress.setCancelButton(None)
        progress.setWindowTitle("Экспорт")
        progress.setMinimumDuration(0)
        completed = 0

        def update(name: str, current: int, split_total: int) -> None:
            nonlocal completed
            before = sum(len(items) for key, items in partitions.items() if list(partitions).index(key) < list(partitions).index(name))
            completed = before + current
            progress.setLabelText(f"{name}: {current} из {split_total}")
            progress.setValue(completed)
            QApplication.processEvents()

        try:
            created = SplitExporter(dataset).export(Path(selected), dialog.values(), update)
        except Exception as error:
            QMessageBox.critical(self, "Ошибка экспорта", str(error))
            return
        finally:
            progress.close()
        for created_path in created:
            self.registry.add(created_path)
        self.datasets_page.refresh()
        QMessageBox.information(self, "Готово", "Созданы датасеты:\n" + "\n".join(str(item) for item in created))

    def dataset_settings(self, path: Path) -> None:
        dataset = LeRobotDataset(path)
        original_count = len([episode for episode in dataset.episodes() if not episode.augmented])
        dialog = AugmentationDialog(
            original_count,
            self,
            default_to_all=True,
            dataset_name=dataset.info.name,
        )
        if not dialog.exec():
            return
        mode, configuration, count = dialog.request()
        title = "Визуальная аугментация датасета" if mode == "visual" else "Аугментация сигналов датасета"
        try:
            created = run_augmentation(self, dataset, mode, configuration, count, title)
        except Exception as error:
            QMessageBox.critical(self, "Ошибка аугментации", str(error))
            return
        self.datasets_page.refresh()
        QMessageBox.information(
            self,
            "Аугментация завершена",
            f"Добавлено эпизодов: {len(created)}. Они отмечены как «Аугментация».",
        )

    def closeEvent(self, event) -> None:  # noqa: N802
        self.episode_page.stop()
        for window in list(self.job_windows):
            if window.process.state() != QProcess.NotRunning:
                window.process.terminate()
        super().closeEvent(event)
