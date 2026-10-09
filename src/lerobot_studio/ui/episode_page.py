from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer, QVideoSink
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from lerobot_studio.domain import Episode, VideoSegment
from lerobot_studio.services.lerobot import LeRobotDataset
from lerobot_studio.ui.dialogs import AugmentationDialog, GraphDialog, MetadataDialog
from lerobot_studio.ui.video_surface import SoftwareVideoSurface
from lerobot_studio.ui.workers import run_augmentation


class CameraPane(QFrame):
    def __init__(self, segment: VideoSegment, primary: bool, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.segment = segment
        self.setObjectName("cameraPane")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        title = QLabel(segment.key)
        title.setObjectName("cameraTitle")
        layout.addWidget(title)
        self.video = SoftwareVideoSurface()
        layout.addWidget(self.video, 1)
        self.player = QMediaPlayer(self)
        self.video_sink = QVideoSink(self)
        self.video_sink.videoFrameChanged.connect(
            self.video.present,
            Qt.ConnectionType.QueuedConnection,
        )
        self.audio = QAudioOutput(self)
        self.audio.setMuted(not primary)
        self.player.setAudioOutput(self.audio)
        self.player.setVideoSink(self.video_sink)
        self.player.setSource(QUrl.fromLocalFile(str(segment.path)))


class EpisodePage(QWidget):
    back_requested = Signal()
    dataset_changed = Signal()

    SPEEDS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.dataset: LeRobotDataset | None = None
        self.episodes: list[Episode] = []
        self.current: Episode | None = None
        self.panes: list[CameraPane] = []
        self.graph_dialogs: list[GraphDialog] = []
        self.trash_dir: Path | None = None
        self._seeking = False
        self._build_ui()
        self._build_shortcuts()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        bar = QHBoxLayout()
        back = QPushButton("‹ Датасеты")
        back.clicked.connect(self.back_requested)
        self.dataset_title = QLabel()
        self.dataset_title.setObjectName("sectionTitle")
        self.episode_box = QComboBox()
        self.episode_box.setMinimumWidth(150)
        self.episode_box.currentIndexChanged.connect(self._episode_selected)
        previous = QPushButton("◀")
        previous.setToolTip("Предыдущий эпизод (Shift+←)")
        previous.clicked.connect(self.previous_episode)
        following = QPushButton("▶")
        following.setToolTip("Следующий эпизод (Shift+→)")
        following.clicked.connect(self.next_episode)
        edit = QPushButton("Редактировать")
        edit.clicked.connect(self.edit_episode)
        graphs = QPushButton("Графики")
        graphs.clicked.connect(self.show_graphs)
        augment = QPushButton("＋ Аугментации")
        augment.clicked.connect(self.augment)
        delete = QPushButton("Удалить")
        delete.setObjectName("dangerButton")
        delete.clicked.connect(self.delete_episode)
        bar.addWidget(back)
        bar.addWidget(self.dataset_title)
        bar.addStretch()
        bar.addWidget(QLabel("Эпизод"))
        bar.addWidget(previous)
        bar.addWidget(self.episode_box)
        bar.addWidget(following)
        bar.addWidget(edit)
        bar.addWidget(graphs)
        bar.addWidget(augment)
        bar.addWidget(delete)
        root.addLayout(bar)
        self.prompt = QLabel()
        self.prompt.setObjectName("promptLabel")
        self.prompt.setWordWrap(True)
        root.addWidget(self.prompt)
        middle = QHBoxLayout()
        self.camera_host = QWidget()
        self.camera_grid = QGridLayout(self.camera_host)
        self.camera_grid.setContentsMargins(0, 0, 0, 0)
        middle.addWidget(self.camera_host, 1)
        self.help_card = self._help_card()
        middle.addWidget(self.help_card)
        root.addLayout(middle, 1)
        controls = QHBoxLayout()
        self.play_button = QPushButton("▶")
        self.play_button.setObjectName("playButton")
        self.play_button.clicked.connect(self.toggle_playback)
        self.position = QSlider(Qt.Horizontal)
        self.position.setRange(0, 1000)
        self.position.sliderPressed.connect(lambda: setattr(self, "_seeking", True))
        self.position.sliderReleased.connect(self._slider_released)
        self.time_label = QLabel("00:00 / 00:00")
        self.mute_button = QPushButton("🔊")
        self.mute_button.clicked.connect(self.toggle_mute)
        self.volume = QSlider(Qt.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setValue(80)
        self.volume.setMaximumWidth(120)
        self.volume.valueChanged.connect(self.set_volume)
        self.speed = QComboBox()
        for value in self.SPEEDS:
            self.speed.addItem(f"{value:g}×", value)
        self.speed.setCurrentIndex(self.SPEEDS.index(1.0))
        self.speed.currentIndexChanged.connect(self.set_speed)
        controls.addWidget(self.play_button)
        controls.addWidget(self.position, 1)
        controls.addWidget(self.time_label)
        controls.addWidget(self.mute_button)
        controls.addWidget(self.volume)
        controls.addWidget(QLabel("Скорость"))
        controls.addWidget(self.speed)
        root.addLayout(controls)

    def _help_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("helpCard")
        card.setFixedWidth(235)
        layout = QVBoxLayout(card)
        title = QLabel("Горячие клавиши")
        title.setObjectName("helpTitle")
        layout.addWidget(title)
        rows = [
            ("Пробел", "Пауза / пуск"),
            ("← / →", "−/+ 1 секунда"),
            ("Shift + ←/→", "Эпизод назад/вперёд"),
            ("↑ / ↓", "Громкость"),
            ("M", "Звук"),
            ("[ / ]", "Скорость"),
            ("G", "Графики"),
            ("E", "Редактировать"),
            ("Delete", "Удалить"),
            ("?", "Скрыть подсказку"),
        ]
        for key, action in rows:
            row = QHBoxLayout()
            key_label = QLabel(key)
            key_label.setObjectName("keyLabel")
            row.addWidget(key_label)
            row.addStretch()
            row.addWidget(QLabel(action))
            layout.addLayout(row)
        layout.addStretch()
        return card

    def _build_shortcuts(self) -> None:
        bindings = {
            "Space": self.toggle_playback,
            "Left": lambda: self.seek_relative(-1000),
            "Right": lambda: self.seek_relative(1000),
            "Shift+Left": self.previous_episode,
            "Shift+Right": self.next_episode,
            "Up": lambda: self.volume.setValue(min(100, self.volume.value() + 5)),
            "Down": lambda: self.volume.setValue(max(0, self.volume.value() - 5)),
            "M": self.toggle_mute,
            "G": self.show_graphs,
            "E": self.edit_episode,
            "Delete": self.delete_episode,
            "[": lambda: self.change_speed(-1),
            "]": lambda: self.change_speed(1),
            "?": lambda: self.help_card.setVisible(not self.help_card.isVisible()),
        }
        self.shortcuts = []
        for sequence, callback in bindings.items():
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(callback)
            self.shortcuts.append(shortcut)

    def open_dataset(self, dataset: LeRobotDataset) -> None:
        self.stop()
        self.dataset = dataset
        self.dataset_title.setText(dataset.info.name)
        self.episodes = dataset.episodes(refresh=True)
        self.episode_box.blockSignals(True)
        self.episode_box.clear()
        for episode in self.episodes:
            suffix = " · Аугментация" if episode.augmented else ""
            self.episode_box.addItem(f"{episode.index:06d}{suffix}", episode.index)
        self.episode_box.blockSignals(False)
        if self.episodes:
            self.episode_box.setCurrentIndex(0)
            self.load_episode(0)
        else:
            self.current = None
            self.prompt.setText("В датасете нет активных эпизодов")
            self._clear_cameras()

    def _episode_selected(self, position: int) -> None:
        if position >= 0 and position < len(self.episodes):
            self.load_episode(position)

    def load_episode(self, position: int) -> None:
        self.stop()
        self.current = self.episodes[position]
        if self.episode_box.currentIndex() != position:
            self.episode_box.blockSignals(True)
            self.episode_box.setCurrentIndex(position)
            self.episode_box.blockSignals(False)
        source = f" · Аугментация · источник {self.current.source_index}" if self.current.augmented else ""
        self.prompt.setText(f"{self.current.prompt or 'Без prompt'}{source} · {self.current.length} кадров")
        self._clear_cameras()
        available = [segment for segment in self.current.videos if segment.path.exists()]
        audio_index = next((index for index, segment in enumerate(available) if segment.has_audio), 0)
        for index, segment in enumerate(available):
            pane = CameraPane(segment, primary=index == audio_index)
            self.camera_grid.addWidget(pane, index // 2, index % 2)
            pane.player.setPlaybackRate(float(self.speed.currentData()))
            pane.audio.setVolume(self.volume.value() / 100)
            self.panes.append(pane)
        if not available:
            message = QLabel("Видео эпизода не найдено или ещё загружается")
            message.setAlignment(Qt.AlignCenter)
            self.camera_grid.addWidget(message, 0, 0)
        if self.panes:
            self.panes[0].player.positionChanged.connect(self._position_changed)
            self.panes[0].player.playbackStateChanged.connect(self._state_changed)
        duration = self._duration_ms()
        self.position.setRange(0, max(1, duration))
        self.position.setValue(0)
        self._update_time(0)
        has_audio = any(segment.has_audio for segment in available)
        self.mute_button.setEnabled(has_audio)
        self.volume.setEnabled(has_audio)
        self.mute_button.setToolTip("В этом эпизоде нет аудио" if not has_audio else "Включить или выключить звук (M)")

    def _clear_cameras(self) -> None:
        self.panes.clear()
        while self.camera_grid.count():
            item = self.camera_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _duration_ms(self) -> int:
        if not self.current:
            return 0
        durations = [int(((item.end_s - item.start_s) if item.end_s is not None else self.current.duration) * 1000) for item in self.current.videos]
        return min(durations) if durations else int(self.current.duration * 1000)

    def toggle_playback(self) -> None:
        if not self.panes:
            return
        if self.panes[0].player.playbackState() == QMediaPlayer.PlayingState:
            for pane in self.panes:
                pane.player.pause()
        else:
            if self.position.value() >= self._duration_ms() - 50:
                self.seek(0)
            for pane in self.panes:
                expected = int(pane.segment.start_s * 1000) + self.position.value()
                pane.player.setPosition(expected)
                pane.player.play()

    def stop(self) -> None:
        for pane in self.panes:
            pane.player.stop()

    def seek(self, relative_ms: int) -> None:
        relative_ms = max(0, min(self._duration_ms(), relative_ms))
        for pane in self.panes:
            pane.player.setPosition(int(pane.segment.start_s * 1000) + relative_ms)
        self.position.setValue(relative_ms)
        self._update_time(relative_ms)

    def seek_relative(self, delta: int) -> None:
        self.seek(self.position.value() + delta)

    def _slider_released(self) -> None:
        self._seeking = False
        self.seek(self.position.value())

    def _position_changed(self, absolute: int) -> None:
        if not self.panes:
            return
        relative = absolute - int(self.panes[0].segment.start_s * 1000)
        if relative >= self._duration_ms():
            for pane in self.panes:
                pane.player.pause()
            relative = self._duration_ms()
        if not self._seeking:
            self.position.setValue(max(0, relative))
        self._update_time(max(0, relative))

    def _state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        self.play_button.setText("Ⅱ" if state == QMediaPlayer.PlayingState else "▶")

    def _update_time(self, position: int) -> None:
        self.time_label.setText(f"{self._clock(position)} / {self._clock(self._duration_ms())}")

    @staticmethod
    def _clock(milliseconds: int) -> str:
        seconds = max(0, milliseconds // 1000)
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    def set_speed(self) -> None:
        value = float(self.speed.currentData())
        for pane in self.panes:
            pane.player.setPlaybackRate(value)

    def change_speed(self, delta: int) -> None:
        self.speed.setCurrentIndex(max(0, min(self.speed.count() - 1, self.speed.currentIndex() + delta)))

    def set_volume(self, value: int) -> None:
        output = self._audio_output()
        if output:
            output.setVolume(value / 100)

    def toggle_mute(self) -> None:
        output = self._audio_output()
        if output is None or not self.mute_button.isEnabled():
            return
        output.setMuted(not output.isMuted())
        self.mute_button.setText("🔇" if output.isMuted() else "🔊")

    def _audio_output(self) -> QAudioOutput | None:
        if not self.panes:
            return None
        pane = next((item for item in self.panes if item.segment.has_audio), self.panes[0])
        return pane.audio

    def previous_episode(self) -> None:
        if self.episodes:
            self.load_episode(max(0, self.episode_box.currentIndex() - 1))

    def next_episode(self) -> None:
        if self.episodes:
            self.load_episode(min(len(self.episodes) - 1, self.episode_box.currentIndex() + 1))

    def show_graphs(self) -> None:
        if not self.dataset or not self.current:
            return
        try:
            dialog = GraphDialog(self.dataset, self.current, self)
        except Exception as error:
            QMessageBox.critical(self, "Не удалось построить графики", str(error))
            return
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        dialog.destroyed.connect(lambda: self._forget_graph_dialog(dialog))
        self.graph_dialogs.append(dialog)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _forget_graph_dialog(self, dialog: GraphDialog) -> None:
        if dialog in self.graph_dialogs:
            self.graph_dialogs.remove(dialog)

    def edit_episode(self) -> None:
        if not self.dataset or not self.current:
            return
        was_playing = self._pause_players()
        dialog = MetadataDialog(self.current, self)
        if dialog.exec():
            prompt, metadata = dialog.values()
            index = self.current.index
            self.dataset.edit_episode(index, prompt, metadata)
            self.open_dataset(self.dataset)
            self._select_index(index)
            self.dataset_changed.emit()
        else:
            self._resume_players(was_playing)

    def augment(self) -> None:
        if not self.dataset:
            return
        was_playing = self._pause_players()
        original_count = len([item for item in self.episodes if not item.augmented])
        dialog = AugmentationDialog(original_count, self)
        if not dialog.exec():
            self._resume_players(was_playing)
            return
        self.stop()
        mode, configuration, count = dialog.request()
        title = "Визуальная аугментация" if mode == "visual" else "Аугментация сигналов"
        try:
            created = run_augmentation(self, self.dataset, mode, configuration, count, title)
        except Exception as error:
            QMessageBox.critical(self, "Ошибка аугментации", str(error))
            self._resume_players(was_playing)
            return
        self.open_dataset(self.dataset)
        if created:
            self._select_index(created[0])
        self.dataset_changed.emit()
        QMessageBox.information(self, "Готово", f"Добавлено эпизодов: {len(created)}")

    def _pause_players(self) -> bool:
        was_playing = bool(self.panes and self.panes[0].player.playbackState() == QMediaPlayer.PlayingState)
        for pane in self.panes:
            pane.player.pause()
        return was_playing

    def _resume_players(self, should_resume: bool) -> None:
        if not should_resume or not self.panes:
            return
        for pane in self.panes:
            pane.player.play()

    def delete_episode(self) -> None:
        if not self.dataset or not self.current:
            return
        if QMessageBox.question(self, "Удалить эпизод", f"Переместить эпизод {self.current.index} в папку удалённых?") != QMessageBox.Yes:
            return
        if self.trash_dir is None:
            selected = QFileDialog.getExistingDirectory(self, "Выберите папку для удалённых эпизодов")
            if not selected:
                return
            self.trash_dir = Path(selected)
        try:
            destination = self.dataset.soft_delete(self.current, self.trash_dir)
        except Exception as error:
            QMessageBox.critical(self, "Не удалось удалить эпизод", str(error))
            return
        self.open_dataset(self.dataset)
        self.dataset_changed.emit()
        QMessageBox.information(self, "Эпизод удалён", f"Копия для восстановления: {destination}")

    def _select_index(self, episode_index: int) -> None:
        for position, episode in enumerate(self.episodes):
            if episode.index == episode_index:
                self.load_episode(position)
                return
