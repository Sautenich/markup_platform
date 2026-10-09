from __future__ import annotations

import json
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressDialog,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from lerobot_studio.domain import Episode
from lerobot_studio.services.augment import AUGMENTATIONS, VISUAL_AUGMENTATIONS, VisualAugmentationSpec
from lerobot_studio.services.lerobot import LeRobotDataset


class MetadataDialog(QDialog):
    def __init__(self, episode: Episode, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Эпизод {episode.index}: prompt и метаданные")
        self.resize(680, 520)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Prompt / инструкция эпизода"))
        self.prompt = QPlainTextEdit(episode.prompt)
        self.prompt.setMaximumHeight(100)
        layout.addWidget(self.prompt)
        layout.addWidget(QLabel("Пользовательские метаданные (JSON)"))
        hidden = {"fps", "augmentation", "augmentation_parameters"}
        editable = {key: value for key, value in episode.metadata.items() if key not in hidden and "/" not in key}
        self.metadata = QPlainTextEdit(json.dumps(editable, indent=2, ensure_ascii=False))
        layout.addWidget(self.metadata)
        self.error = QLabel()
        self.error.setStyleSheet("color: #ff7b72")
        layout.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _validate(self) -> None:
        try:
            value = json.loads(self.metadata.toPlainText() or "{}")
            if not isinstance(value, dict):
                raise ValueError("Ожидается JSON-объект")
            self._value = value
        except (json.JSONDecodeError, ValueError) as error:
            self.error.setText(str(error))
            return
        self.accept()

    def values(self) -> tuple[str, dict[str, Any]]:
        return self.prompt.toPlainText().strip(), self._value


class VisualParameterRow(QWidget):
    def __init__(self, spec: VisualAugmentationSpec, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.spec = spec
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        self.enabled = QCheckBox(spec.title)
        self.enabled.setToolTip(spec.description)
        self.enabled.setMinimumWidth(175)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, round((spec.maximum - spec.minimum) / spec.step))
        self.spin = QDoubleSpinBox()
        decimals = max(0, len(f"{spec.step:.8f}".rstrip("0").split(".")[-1])) if spec.step < 1 else 0
        self.spin.setDecimals(decimals)
        self.spin.setRange(spec.minimum, spec.maximum)
        self.spin.setSingleStep(spec.step)
        self.spin.setValue(spec.default)
        self.spin.setSuffix(f" {spec.unit}" if spec.unit else "")
        self.spin.setMinimumWidth(105)
        self.slider.setValue(round((spec.default - spec.minimum) / spec.step))
        self.slider.setToolTip(f"{spec.title}: {spec.description}")
        self.spin.setToolTip("Можно ввести точное значение вручную")
        self.slider.sliderPressed.connect(lambda: self.enabled.setChecked(True))
        self.slider.valueChanged.connect(self._slider_changed)
        self.spin.valueChanged.connect(self._spin_changed)
        layout.addWidget(self.enabled)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)

    def _slider_changed(self, value: int) -> None:
        self.enabled.setChecked(True)
        self.spin.blockSignals(True)
        self.spin.setValue(self.spec.minimum + value * self.spec.step)
        self.spin.blockSignals(False)

    def _spin_changed(self, value: float) -> None:
        self.enabled.setChecked(True)
        self.slider.blockSignals(True)
        self.slider.setValue(round((value - self.spec.minimum) / self.spec.step))
        self.slider.blockSignals(False)

    def selected_value(self) -> float | None:
        return self.spin.value() if self.enabled.isChecked() else None


class AugmentationDialog(QDialog):
    def __init__(
        self,
        episode_count: int,
        parent: QWidget | None = None,
        *,
        default_to_all: bool = False,
        dataset_name: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Настройки датасета — {dataset_name}" if dataset_name else "Добавить аугментированные эпизоды")
        self.resize(780, 690)
        layout = QVBoxLayout(self)
        if dataset_name:
            heading = QLabel(
                f"Аугментация всего датасета: {episode_count} исходных эпизодов. "
                "По умолчанию будет создано по одному новому эпизоду для каждого исходного."
            )
            heading.setWordWrap(True)
            layout.addWidget(heading)
        self.tabs = QTabWidget()
        signal_tab = QWidget()
        signal_layout = QFormLayout(signal_tab)
        self.kind = QComboBox()
        for spec in AUGMENTATIONS:
            self.kind.addItem(spec.title, spec.key)
            self.kind.setItemData(self.kind.count() - 1, spec.description, Qt.ToolTipRole)
        signal_layout.addRow("Аугментация сигналов", self.kind)
        signal_note = QLabel("Изменяет состояния суставов, действия или временную структуру. Исходное видео используется без изменений.")
        signal_note.setWordWrap(True)
        signal_layout.addRow(signal_note)
        self.tabs.addTab(signal_tab, "Сигналы")

        visual_tab = QWidget()
        visual_layout = QVBoxLayout(visual_tab)
        visual_note = QLabel("Включите один или несколько эффектов. Они будут применены ко всем камерам выбранных эпизодов.")
        visual_note.setWordWrap(True)
        visual_layout.addWidget(visual_note)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        rows_host = QWidget()
        rows_layout = QVBoxLayout(rows_host)
        self.visual_rows: dict[str, VisualParameterRow] = {}
        for spec in VISUAL_AUGMENTATIONS:
            row = VisualParameterRow(spec)
            self.visual_rows[spec.key] = row
            rows_layout.addWidget(row)
        rows_layout.addStretch()
        scroll.setWidget(rows_host)
        visual_layout.addWidget(scroll)
        self.tabs.addTab(visual_tab, "Видео · 10 эффектов")
        layout.addWidget(self.tabs, 1)

        count_form = QFormLayout()
        self.count = QSpinBox()
        self.count.setRange(1, 100_000)
        self.count.setValue(episode_count if default_to_all else min(episode_count, 10))
        count_form.addRow("Новых эпизодов", self.count)
        note = QLabel(
            "До количества исходных эпизодов источники не повторяются. "
            "При большем количестве каждый исходный будет использован хотя бы один раз, затем источники выбираются случайно."
        )
        note.setWordWrap(True)
        count_form.addRow(note)
        layout.addLayout(count_form)
        self.error = QLabel()
        self.error.setStyleSheet("color: #ff7b72")
        layout.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _validate(self) -> None:
        if self.tabs.currentIndex() == 1 and not self.visual_values():
            self.error.setText("Включите хотя бы один визуальный эффект")
            return
        self.accept()

    def visual_values(self) -> dict[str, float]:
        return {
            key: value
            for key, row in self.visual_rows.items()
            if (value := row.selected_value()) is not None
        }

    def request(self) -> tuple[str, str | dict[str, float], int]:
        if self.tabs.currentIndex() == 1:
            return "visual", self.visual_values(), self.count.value()
        return "signals", str(self.kind.currentData()), self.count.value()


class SplitDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Создать split-датасеты")
        layout = QFormLayout(self)
        self.train = self._percent(80)
        self.validation = self._percent(10)
        self.test = self._percent(10)
        layout.addRow("Train, %", self.train)
        layout.addRow("Validation, %", self.validation)
        layout.addRow("Test, %", self.test)
        self.status = QLabel()
        layout.addRow(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    @staticmethod
    def _percent(value: int) -> QSpinBox:
        widget = QSpinBox()
        widget.setRange(0, 100)
        widget.setSuffix(" %")
        widget.setValue(value)
        return widget

    def _validate(self) -> None:
        if sum(self.values()) != 100:
            self.status.setText("Сумма должна быть равна 100%")
            self.status.setStyleSheet("color: #ff7b72")
            return
        self.accept()

    def values(self) -> tuple[int, int, int]:
        return self.train.value(), self.validation.value(), self.test.value()


class GraphLoadSignals(QObject):
    ready = Signal(object)
    failed = Signal(str)


class GraphLoadWorker(QRunnable):
    def __init__(self, dataset: LeRobotDataset, episode: Episode) -> None:
        super().__init__()
        self.dataset = dataset
        self.episode = episode
        self.signals = GraphLoadSignals()

    @Slot()
    def run(self) -> None:
        try:
            frame = self.dataset.dataframe(self.episode)
            time = (
                frame["timestamp"].to_numpy()
                if "timestamp" in frame
                else np.arange(len(frame)) / self.dataset.info.fps
            )
            joint_keys = [key for key in ("observation.state", "action") if key in frame]
            payload = {
                "time": time,
                "joints": self._series(frame, joint_keys),
                "imu": self._series(frame, self.dataset.info.imu_keys),
            }
        except Exception as error:
            self.signals.failed.emit(str(error))
            return
        self.signals.ready.emit(payload)

    def _series(self, frame, keys: list[str]) -> list[tuple[str, np.ndarray, list[str]]]:
        result = []
        for key in keys:
            if key not in frame or frame.empty:
                continue
            values = (
                np.stack(frame[key].to_numpy())
                if np.ndim(frame[key].iloc[0])
                else frame[key].to_numpy()[:, None]
            )
            names = self.dataset.info.features.get(key, {}).get("names")
            if isinstance(names, list) and len(names) == 1 and isinstance(names[0], list):
                names = names[0]
            if not isinstance(names, list):
                names = [f"{key}[{index}]" for index in range(values.shape[1])]
            result.append((key, values, [str(name) for name in names]))
        return result


class GraphDialog(QDialog):
    def __init__(self, dataset: LeRobotDataset, episode: Episode, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Сигналы эпизода {episode.index}")
        self.resize(1100, 720)
        self.setModal(False)
        self.loaded = False
        self.layout = QVBoxLayout(self)
        self.loading = QLabel("Загрузка сигналов…")
        self.loading.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.loading, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        self.layout.addWidget(buttons)
        self.worker = GraphLoadWorker(dataset, episode)
        self.worker.signals.ready.connect(self._show_payload)
        self.worker.signals.failed.connect(self._show_error)
        QThreadPool.globalInstance().start(self.worker)

    @Slot(object)
    def _show_payload(self, payload: dict[str, Any]) -> None:
        if self.loading is None:
            return
        self.layout.removeWidget(self.loading)
        self.loading.deleteLater()
        self.loading = None
        tabs = QTabWidget()
        tabs.addTab(
            self._plot_group(payload["time"], payload["joints"], "Нет данных о суставах"),
            "Суставы",
        )
        tabs.addTab(
            self._plot_group(payload["time"], payload["imu"], "IMU-поля в этом датасете не найдены"),
            "IMU",
        )
        self.layout.insertWidget(0, tabs, 1)
        self.loaded = True

    @Slot(str)
    def _show_error(self, message: str) -> None:
        if self.loading is not None:
            self.loading.setText(f"Не удалось построить графики:\n{message}")
            self.loading.setStyleSheet("color: #ff7b72")

    @staticmethod
    def _plot_group(
        time: np.ndarray,
        series: list[tuple[str, np.ndarray, list[str]]],
        empty_text: str,
    ) -> QWidget:
        if not series:
            label = QLabel(empty_text)
            label.setAlignment(Qt.AlignCenter)
            return label
        tabs = QTabWidget()
        for key, values, names in series:
            plot = pg.PlotWidget()
            plot.setBackground("#111820")
            plot.showGrid(x=True, y=True, alpha=0.25)
            plot.setLabel("bottom", "Время", units="s")
            plot.setLabel("left", key)
            plot.addLegend(offset=(10, 10))
            plot.getPlotItem().setDownsampling(auto=True, mode="peak")
            plot.getPlotItem().setClipToView(True)
            for column in range(values.shape[1]):
                plot.plot(time, values[:, column], pen=pg.intColor(column, values.shape[1]), name=str(names[column] if column < len(names) else column))
            tabs.addTab(plot, key)
        return tabs


class OperationProgress(QProgressDialog):
    def __init__(self, title: str, maximum: int, parent: QWidget | None = None) -> None:
        super().__init__("Подготовка…", "", 0, maximum, parent)
        self.setWindowTitle(title)
        self.setCancelButton(None)
        self.setWindowModality(Qt.WindowModal)
        self.setMinimumDuration(0)
        self.setAutoClose(False)
        self.setAutoReset(False)
