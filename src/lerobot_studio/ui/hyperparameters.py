from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from lerobot_studio.services.model_jobs import MODEL_NAMES, ProfileStore


PARAMETER_LABELS = {
    "python_executable": "Python окружения модели",
    "base_model_path": "Базовая модель",
    "embodiment_tag": "Embodiment tag",
    "modality_config_path": "Modality config",
    "mllm_path": "MLLM backbone",
    "expert_path": "Action Expert",
    "resume_from": "Базовый checkpoint",
    "data_config_name": "Data config",
    "max_steps": "Шагов обучения",
    "global_batch_size": "Global batch size",
    "learning_rate": "Learning rate",
    "weight_decay": "Weight decay",
    "warmup_ratio": "Warmup ratio",
    "save_steps": "Сохранять каждые N шагов",
    "action_chunk_length": "Action chunk length",
    "num_gpus": "Количество GPU",
    "dataloader_num_workers": "DataLoader workers",
    "num_workers": "DataLoader workers",
}


class HyperparameterDialog(QDialog):
    def __init__(
        self,
        model: str,
        values: dict[str, Any] | None = None,
        profile_store: ProfileStore | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.model = model
        self.store = profile_store or ProfileStore()
        self.editors: dict[str, QLineEdit] = {}
        self.types: dict[str, type] = {}
        self.setWindowTitle(f"Гиперпараметры · {MODEL_NAMES[model]}")
        self.resize(640, 680)
        root = QVBoxLayout(self)
        profile_row = QHBoxLayout()
        profile_row.addWidget(QLabel("Профиль"))
        self.profile = QComboBox()
        self._reload_profiles()
        profile_row.addWidget(self.profile, 1)
        save = QPushButton("Сохранить текущие как профиль")
        save.clicked.connect(self.save_profile)
        profile_row.addWidget(save)
        root.addLayout(profile_row)
        hint = QLabel("Выберите готовый профиль или измените любое поле вручную.")
        hint.setObjectName("subtitle")
        root.addWidget(hint)
        body = QWidget()
        self.form = QFormLayout(body)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)
        root.addWidget(scroll, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self.profile.currentTextChanged.connect(self._profile_selected)
        initial = values or self.store.profiles(model)[self.profile.currentText()]
        matching_profile = next(
            (name for name, profile_values in self.store.profiles(model).items() if profile_values == initial),
            None,
        )
        self.profile.setCurrentText(matching_profile) if matching_profile else self.profile.setCurrentIndex(-1)
        self._build_fields(initial)

    def _reload_profiles(self, selected: str | None = None) -> None:
        self.profile.blockSignals(True)
        self.profile.clear()
        self.profile.addItems(self.store.profiles(self.model))
        if selected:
            self.profile.setCurrentText(selected)
        self.profile.blockSignals(False)

    def _build_fields(self, values: dict[str, Any]) -> None:
        while self.form.rowCount():
            self.form.removeRow(0)
        self.editors.clear()
        self.types.clear()
        for key, value in values.items():
            editor = QLineEdit(str(value))
            editor.textEdited.connect(lambda _text: self.profile.setCurrentIndex(-1))
            self.editors[key] = editor
            self.types[key] = type(value)
            self.form.addRow(PARAMETER_LABELS.get(key, key), editor)

    def _profile_selected(self, name: str) -> None:
        if name:
            self._build_fields(self.store.profiles(self.model)[name])

    def values(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, editor in self.editors.items():
            raw = editor.text().strip()
            expected = self.types[key]
            if expected is bool:
                result[key] = raw.lower() in {"1", "true", "yes", "да"}
            else:
                result[key] = expected(raw)
        return result

    def _accept_if_valid(self) -> None:
        try:
            values = self.values()
            if int(values.get("max_steps", 1)) < 1 or int(values.get("num_gpus", 1)) < 1:
                raise ValueError("Количество шагов и GPU должно быть больше нуля")
        except (TypeError, ValueError) as error:
            QMessageBox.warning(self, "Некорректные параметры", str(error))
            return
        self.accept()

    def save_profile(self) -> None:
        try:
            values = self.values()
        except (TypeError, ValueError) as error:
            QMessageBox.warning(self, "Некорректные параметры", str(error))
            return
        name, accepted = QInputDialog.getText(self, "Новый профиль", "Название профиля")
        name = name.strip()
        if not accepted or not name:
            return
        self.store.save(self.model, name, values)
        self._reload_profiles(name)
