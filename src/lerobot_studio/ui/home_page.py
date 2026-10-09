from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


class HomePage(QWidget):
    datasets_requested = Signal()
    training_requested = Signal()
    testing_requested = Signal()

    def __init__(self, dry_run: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(48, 40, 48, 48)
        title = QLabel("LeRobot Studio")
        title.setObjectName("pageTitle")
        subtitle = QLabel("Единое рабочее пространство для данных и VLA-моделей")
        subtitle.setObjectName("subtitle")
        root.addWidget(title)
        root.addWidget(subtitle)
        root.addStretch(1)
        cards = QHBoxLayout()
        cards.setSpacing(22)
        cards.addWidget(
            self._card(
                "01",
                "Работа с датасетами",
                "Просмотр и разметка эпизодов, аугментации, метаданные и train/validation/test сплиты.",
                "Открыть датасеты",
                self.datasets_requested.emit,
            )
        )
        cards.addWidget(
            self._card(
                "02",
                "Обучение моделей",
                "Выбор GR00T или Being-H0.5, нескольких датасетов и профиля гиперпараметров.",
                "Настроить обучение",
                self.training_requested.emit,
            )
        )
        cards.addWidget(
            self._card(
                "03",
                "Тестирование моделей",
                "Open-loop проверка checkpoint на датасете с Action MAE / RMSE и мониторингом ресурсов.",
                "Настроить тест",
                self.testing_requested.emit,
            )
        )
        root.addLayout(cards, 3)
        root.addStretch(1)

    @staticmethod
    def _card(number: str, title: str, description: str, button_text: str, callback) -> QFrame:
        card = QFrame()
        card.setObjectName("menuCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)
        marker = QLabel(number)
        marker.setObjectName("menuNumber")
        heading = QLabel(title)
        heading.setObjectName("menuTitle")
        description_label = QLabel(description)
        description_label.setObjectName("menuDescription")
        description_label.setWordWrap(True)
        layout.addWidget(marker)
        layout.addWidget(heading)
        layout.addWidget(description_label)
        layout.addStretch()
        button = QPushButton(button_text)
        button.setObjectName("primaryButton")
        button.clicked.connect(callback)
        layout.addWidget(button)
        return card
