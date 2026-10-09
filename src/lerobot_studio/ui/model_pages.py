from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from lerobot_studio.config import DatasetRegistry
from lerobot_studio.services.model_jobs import (
    BEING_H05,
    GR00T,
    MODEL_NAMES,
    ProfileStore,
    build_test_job,
    build_train_job,
    checkpoint_model,
    default_output_dir,
    discover_checkpoints,
)
from lerobot_studio.ui.hyperparameters import HyperparameterDialog


class DatasetChecklist(QListWidget):
    def __init__(self, registry: DatasetRegistry, multiple: bool, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.registry = registry
        self.multiple = multiple
        self.setMinimumHeight(210)
        self.setCursor(Qt.PointingHandCursor)
        self.refresh()
        self.itemChanged.connect(self._enforce_single)

    def refresh(self) -> None:
        self.clear()
        for path in self.registry.load():
            item = QListWidgetItem(f"{path.name}\n{path}")
            item.setData(Qt.UserRole, str(path))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            item.setCheckState(Qt.Unchecked)
            item.setToolTip("Нажмите на строку, чтобы выбрать датасет")
            self.addItem(item)

    def _toggle(self, item: QListWidgetItem) -> None:
        item.setCheckState(Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked)
        self.setCurrentItem(item)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        item = self.itemAt(event.position().toPoint())
        if item is None:
            super().mousePressEvent(event)
            return
        # QListWidget normally toggles a checkbox only when its tiny indicator is
        # clicked.  Here the whole row is an explicit selection target.
        self._toggle(item)
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter) and self.currentItem():
            self._toggle(self.currentItem())
            event.accept()
            return
        super().keyPressEvent(event)

    def _enforce_single(self, changed: QListWidgetItem) -> None:
        if self.multiple or changed.checkState() != Qt.Checked:
            return
        self.blockSignals(True)
        for index in range(self.count()):
            item = self.item(index)
            if item is not changed:
                item.setCheckState(Qt.Unchecked)
        self.blockSignals(False)

    def selected_paths(self) -> list[Path]:
        return [
            Path(self.item(index).data(Qt.UserRole))
            for index in range(self.count())
            if self.item(index).checkState() == Qt.Checked
        ]


class _ModelPage(QWidget):
    home_requested = Signal()

    def __init__(self, title: str, subtitle: str, dry_run: bool, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.dry_run = dry_run
        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(32, 26, 32, 30)
        top = QHBoxLayout()
        back = QPushButton("← Главное меню")
        back.clicked.connect(self.home_requested.emit)
        top.addWidget(back)
        top.addStretch()
        self.root.addLayout(top)
        heading = QLabel(title)
        heading.setObjectName("pageTitle")
        description = QLabel(subtitle)
        description.setObjectName("subtitle")
        self.root.addWidget(heading)
        self.root.addWidget(description)

    @staticmethod
    def _section(title: str) -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame()
        frame.setObjectName("settingsCard")
        layout = QVBoxLayout(frame)
        label = QLabel(title)
        label.setObjectName("sectionTitle")
        layout.addWidget(label)
        return frame, layout


class TrainingPage(_ModelPage):
    job_requested = Signal(object)

    def __init__(self, workspace: Path, registry: DatasetRegistry, dry_run: bool, parent: QWidget | None = None) -> None:
        super().__init__(
            "Обучение VLA-модели",
            "Выберите архитектуру, один или несколько датасетов и конфигурацию запуска.",
            dry_run,
            parent,
        )
        self.workspace = workspace
        self.registry = registry
        self.profile_store = ProfileStore()
        self.hyperparameters: dict[str, object] = {}

        model_card, model_layout = self._section("1. Модель и гиперпараметры")
        model_row = QHBoxLayout()
        self.model = QComboBox()
        for key, name in MODEL_NAMES.items():
            self.model.addItem(name, key)
        self.hyper_button = QPushButton("Открыть гиперпараметры…")
        self.hyper_button.clicked.connect(self.open_hyperparameters)
        model_row.addWidget(QLabel("VLA-модель"))
        model_row.addWidget(self.model, 1)
        model_row.addWidget(self.hyper_button)
        model_layout.addLayout(model_row)
        self.profile_summary = QLabel()
        self.profile_summary.setWordWrap(True)
        self.profile_summary.setObjectName("mutedLabel")
        model_layout.addWidget(self.profile_summary)
        self.root.addWidget(model_card)

        data_card, data_layout = self._section("2. Датасеты для обучения")
        self.datasets = DatasetChecklist(registry, multiple=True)
        data_layout.addWidget(self.datasets)
        self.root.addWidget(data_card, 1)

        output_card, output_layout = self._section("3. Результат")
        output_row = QHBoxLayout()
        self.output = QLineEdit()
        browse = QPushButton("Выбрать папку…")
        browse.clicked.connect(self._browse_output)
        output_row.addWidget(QLabel("Папка checkpoint"))
        output_row.addWidget(self.output, 1)
        output_row.addWidget(browse)
        output_layout.addLayout(output_row)
        self.root.addWidget(output_card)

        actions = QHBoxLayout()
        actions.addStretch()
        start = QPushButton("Начать обучение")
        start.setObjectName("primaryButton")
        start.clicked.connect(self.start)
        actions.addWidget(start)
        self.root.addLayout(actions)
        self.model.currentIndexChanged.connect(self._model_changed)
        self._model_changed()

    def _model_changed(self) -> None:
        model = self.model.currentData()
        profiles = self.profile_store.profiles(model)
        preferred = profiles.get("Сбалансированный") or next(iter(profiles.values()))
        self.hyperparameters = dict(preferred)
        self.output.setText(str(default_output_dir(self.workspace, model)))
        self._update_summary()

    def _update_summary(self) -> None:
        values = self.hyperparameters
        self.profile_summary.setText(
            f"Шаги: {values.get('max_steps', '—')}  ·  LR: {values.get('learning_rate', '—')}  ·  "
            f"GPU: {values.get('num_gpus', '—')}  ·  Сохранение: {values.get('save_steps', '—')}"
        )

    def open_hyperparameters(self) -> None:
        dialog = HyperparameterDialog(
            self.model.currentData(), self.hyperparameters, self.profile_store, self
        )
        if dialog.exec():
            self.hyperparameters = dialog.values()
            self._update_summary()

    def _browse_output(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Папка для checkpoint", self.output.text())
        if selected:
            self.output.setText(selected)

    def start(self) -> None:
        datasets = self.datasets.selected_paths()
        if not datasets:
            QMessageBox.warning(self, "Нет датасетов", "Выберите хотя бы один датасет для обучения.")
            return
        if not self.output.text().strip():
            QMessageBox.warning(self, "Нет папки", "Укажите папку для сохранения checkpoint.")
            return
        spec = build_train_job(
            self.workspace,
            self.model.currentData(),
            datasets,
            Path(self.output.text()),
            self.hyperparameters,
            self.dry_run,
        )
        self.job_requested.emit(spec)


class TestingPage(_ModelPage):
    job_requested = Signal(object)

    def __init__(self, workspace: Path, registry: DatasetRegistry, dry_run: bool, parent: QWidget | None = None) -> None:
        super().__init__(
            "Тестирование checkpoint",
            "Запустите open-loop оценку на выбранном датасете и следите за Action MAE / RMSE.",
            dry_run,
            parent,
        )
        self.workspace = workspace
        checkpoint_card, checkpoint_layout = self._section("1. Checkpoint")
        form = QFormLayout()
        self.model = QComboBox()
        for key, name in MODEL_NAMES.items():
            self.model.addItem(name, key)
        self.checkpoint = QComboBox()
        self.checkpoint.setEditable(True)
        browse = QPushButton("Выбрать…")
        browse.clicked.connect(self._browse_checkpoint)
        checkpoint_row = QHBoxLayout()
        checkpoint_row.addWidget(self.checkpoint, 1)
        checkpoint_row.addWidget(browse)
        form.addRow("Модель", self.model)
        form.addRow("Путь", checkpoint_row)
        checkpoint_layout.addLayout(form)
        self.root.addWidget(checkpoint_card)

        data_card, data_layout = self._section("2. Датасет для теста")
        self.datasets = DatasetChecklist(registry, multiple=False)
        data_layout.addWidget(self.datasets)
        self.root.addWidget(data_card, 1)

        actions = QHBoxLayout()
        actions.addStretch()
        start = QPushButton("Начать тестирование")
        start.setObjectName("primaryButton")
        start.clicked.connect(self.start)
        actions.addWidget(start)
        self.root.addLayout(actions)
        self.refresh_checkpoints()
        self.checkpoint.currentTextChanged.connect(self._detect_model)

    def refresh_checkpoints(self) -> None:
        current = self.checkpoint.currentText()
        self.checkpoint.clear()
        self.checkpoint.addItems(str(path) for path in discover_checkpoints(self.workspace))
        if current:
            self.checkpoint.setCurrentText(current)

    def _browse_checkpoint(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Выберите checkpoint", self.checkpoint.currentText())
        if selected:
            self.checkpoint.setCurrentText(selected)

    def _detect_model(self, value: str) -> None:
        if not value:
            return
        detected = checkpoint_model(Path(value))
        if detected:
            index = self.model.findData(detected)
            self.model.setCurrentIndex(index)

    def start(self) -> None:
        datasets = self.datasets.selected_paths()
        if not datasets:
            QMessageBox.warning(self, "Нет датасета", "Выберите датасет для тестирования.")
            return
        checkpoint_text = self.checkpoint.currentText().strip()
        if not checkpoint_text:
            QMessageBox.warning(self, "Нет checkpoint", "Выберите checkpoint модели.")
            return
        checkpoint = Path(checkpoint_text)
        if not self.dry_run and not checkpoint.exists():
            QMessageBox.warning(self, "Checkpoint не найден", f"Путь не существует:\n{checkpoint}")
            return
        output = self.workspace / "test_results" / datetime.now().strftime("%Y%m%d_%H%M%S")
        spec = build_test_job(
            self.workspace,
            self.model.currentData(),
            datasets[0],
            checkpoint,
            output,
            self.dry_run,
        )
        self.job_requested.emit(spec)
