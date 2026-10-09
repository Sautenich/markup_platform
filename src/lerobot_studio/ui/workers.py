from __future__ import annotations

import logging

from PySide6.QtCore import QEventLoop, QObject, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import QWidget

from lerobot_studio.services.augment import AugmentationService
from lerobot_studio.services.lerobot import LeRobotDataset
from lerobot_studio.ui.dialogs import OperationProgress


LOGGER = logging.getLogger(__name__)


class AugmentationWorker(QObject):
    progress = Signal(int, int)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        dataset: LeRobotDataset,
        mode: str,
        configuration: str | dict[str, float],
        count: int,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.mode = mode
        self.configuration = configuration
        self.count = count

    @Slot()
    def run(self) -> None:
        LOGGER.info(
            "Augmentation worker started mode=%s count=%s dataset=%s configuration=%s",
            self.mode,
            self.count,
            self.dataset.root,
            self.configuration,
        )
        try:
            service = AugmentationService(self.dataset)
            if self.mode == "visual":
                created = service.create_visual(dict(self.configuration), self.count, self.progress.emit)
            else:
                created = service.create(str(self.configuration), self.count, self.progress.emit)
        except Exception as error:
            LOGGER.exception("Augmentation worker failed")
            self.failed.emit(f"{error}\n\nПодробный лог: {self.dataset.root / 'studio' / 'debug.log'}")
            return
        LOGGER.info("Augmentation worker completed: %s", created)
        self.finished.emit(created)


class AugmentationUiBridge(QObject):
    """Receive worker signals on the GUI thread.

    PySide can execute signals connected to plain nested Python functions in
    the emitter's thread.  Making the receiver a QObject with GUI-thread
    affinity and using queued connections prevents background QPainter access.
    """

    def __init__(
        self,
        progress: OperationProgress,
        loop: QEventLoop,
        thread: QThread,
        result: list[list[int]],
        errors: list[str],
    ) -> None:
        super().__init__(progress)
        self.progress_dialog = progress
        self.loop = loop
        self.worker_thread = thread
        self.result = result
        self.errors = errors

    @Slot(int, int)
    def update(self, value: int, total: int) -> None:
        self.progress_dialog.setLabelText(f"Эпизод {value} из {total}")
        self.progress_dialog.setMaximum(total)
        self.progress_dialog.setValue(value)

    @Slot(object)
    def complete(self, created: list[int]) -> None:
        self.result.append(created)
        self.worker_thread.quit()
        self.loop.quit()

    @Slot(str)
    def fail(self, message: str) -> None:
        self.errors.append(message)
        self.worker_thread.quit()
        self.loop.quit()


def run_augmentation(
    parent: QWidget,
    dataset: LeRobotDataset,
    mode: str,
    configuration: str | dict[str, float],
    count: int,
    title: str,
) -> list[int]:
    """Run CPU/FFmpeg work off the UI thread while keeping a modal progress view."""
    progress = OperationProgress(title, count, parent)
    thread = QThread(parent)
    worker = AugmentationWorker(dataset, mode, configuration, count)
    worker.moveToThread(thread)
    loop = QEventLoop()
    result: list[list[int]] = []
    errors: list[str] = []
    bridge = AugmentationUiBridge(progress, loop, thread, result, errors)

    thread.started.connect(worker.run)
    worker.progress.connect(bridge.update, Qt.ConnectionType.QueuedConnection)
    worker.finished.connect(bridge.complete, Qt.ConnectionType.QueuedConnection)
    worker.failed.connect(bridge.fail, Qt.ConnectionType.QueuedConnection)
    # Schedule destruction while the worker event loop is still alive.
    worker.finished.connect(worker.deleteLater)
    worker.failed.connect(worker.deleteLater)
    progress.show()
    thread.start()
    loop.exec()
    thread.wait()
    progress.close()
    thread.deleteLater()
    if errors:
        raise RuntimeError(errors[0])
    return result[0] if result else []
