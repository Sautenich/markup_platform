from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lerobot_studio.config import DatasetRegistry
from lerobot_studio.services.lerobot import LeRobotDataset


class DatasetsPage(QWidget):
    home_requested = Signal()
    open_requested = Signal(object)
    split_requested = Signal(object)
    settings_requested = Signal(object)

    def __init__(self, registry: DatasetRegistry, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.registry = registry
        self.paths: list[Path] = []
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 28, 32, 28)
        navigation = QHBoxLayout()
        home = QPushButton("← Главное меню")
        home.clicked.connect(self.home_requested.emit)
        navigation.addWidget(home)
        navigation.addStretch()
        root.addLayout(navigation)
        title = QLabel("Работа с датасетами")
        title.setObjectName("pageTitle")
        subtitle = QLabel("Локальные датасеты для просмотра, редактирования и подготовки выборок")
        subtitle.setObjectName("subtitle")
        root.addWidget(title)
        root.addWidget(subtitle)
        actions = QHBoxLayout()
        add = QPushButton("＋ Добавить датасет")
        add.setObjectName("primaryButton")
        add.clicked.connect(self.add_dataset)
        open_button = QPushButton("Открыть")
        open_button.clicked.connect(self.open_selected)
        split = QPushButton("Создать сплиты")
        split.clicked.connect(self.split_selected)
        settings = QPushButton("Настройки датасета")
        settings.clicked.connect(self.settings_selected)
        forget = QPushButton("Убрать из списка")
        forget.clicked.connect(self.remove_selected)
        actions.addWidget(add)
        actions.addWidget(open_button)
        actions.addWidget(settings)
        actions.addWidget(split)
        actions.addStretch()
        actions.addWidget(forget)
        root.addLayout(actions)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["Датасет", "Робот", "Эпизоды", "Аугментации", "FPS", "Версия", "Путь"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(6, QHeaderView.Stretch)
        self.table.doubleClicked.connect(lambda _index: self.open_selected())
        root.addWidget(self.table)
        tip = QFrame()
        tip.setObjectName("tipCard")
        tip_layout = QVBoxLayout(tip)
        tip_layout.addWidget(QLabel("Как добавить датасет"))
        tip_layout.addWidget(QLabel("Выберите корневую папку, содержащую meta/info.json. Поддерживаются структуры LeRobot v2 и v3."))
        root.addWidget(tip)
        self.refresh()

    def refresh(self) -> None:
        self.paths = self.registry.load()
        self.table.setRowCount(0)
        for path in self.paths:
            try:
                dataset = LeRobotDataset(path)
                episodes = dataset.episodes()
                values = [
                    dataset.info.name,
                    dataset.info.robot_type,
                    str(len(episodes)),
                    str(sum(item.augmented for item in episodes)),
                    f"{dataset.info.fps:g}",
                    dataset.info.version,
                    str(path),
                ]
            except Exception as error:
                values = [path.name, "Ошибка", "—", "—", "—", "—", f"{path}: {error}"]
            row = self.table.rowCount()
            self.table.insertRow(row)
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.UserRole, str(path))
                self.table.setItem(row, column, item)
        if self.table.rowCount():
            self.table.selectRow(0)

    def add_dataset(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Выберите корень LeRobot-датасета")
        if not selected:
            return
        path = Path(selected)
        if not LeRobotDataset.validate(path):
            QMessageBox.warning(self, "Не LeRobot-датасет", "В выбранной папке нет meta/info.json")
            return
        self.registry.add(path)
        self.refresh()

    def selected_path(self) -> Path | None:
        row = self.table.currentRow()
        return Path(self.table.item(row, 0).data(Qt.UserRole)) if row >= 0 else None

    def open_selected(self) -> None:
        path = self.selected_path()
        if path:
            self.open_requested.emit(path)

    def split_selected(self) -> None:
        path = self.selected_path()
        if path:
            self.split_requested.emit(path)

    def settings_selected(self) -> None:
        path = self.selected_path()
        if path:
            self.settings_requested.emit(path)

    def remove_selected(self) -> None:
        path = self.selected_path()
        if path:
            self.registry.remove(path)
            self.refresh()
