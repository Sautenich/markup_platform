from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer, Qt, Signal
from PySide6.QtGui import QCloseEvent, QFont, QTextCursor
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from lerobot_studio.services.model_jobs import JobSpec, MODEL_NAMES


class ProcessResources:
    def __init__(self) -> None:
        self.previous_ticks: int | None = None
        self.previous_time: float | None = None
        self.clock_ticks = os.sysconf("SC_CLK_TCK")

    @staticmethod
    def _process_tree(root_pid: int) -> set[int]:
        pids = {root_pid}
        changed = True
        while changed:
            changed = False
            for entry in Path("/proc").iterdir():
                if not entry.name.isdigit() or int(entry.name) in pids:
                    continue
                try:
                    status = (entry / "status").read_text(encoding="utf-8")
                    parent = int(re.search(r"^PPid:\s+(\d+)", status, re.MULTILINE).group(1))
                except (OSError, AttributeError, ValueError):
                    continue
                if parent in pids:
                    pids.add(int(entry.name))
                    changed = True
        return pids

    @staticmethod
    def _ticks(pids: set[int]) -> int:
        total = 0
        for pid in pids:
            try:
                fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
                total += int(fields[13]) + int(fields[14])
            except (OSError, IndexError, ValueError):
                pass
        return total

    def sample(self, root_pid: int) -> tuple[str, str]:
        pids = self._process_tree(root_pid)
        now = time.monotonic()
        ticks = self._ticks(pids)
        cpu = "—"
        if self.previous_ticks is not None and self.previous_time is not None:
            elapsed = now - self.previous_time
            if elapsed > 0:
                cpu = f"{(ticks - self.previous_ticks) / self.clock_ticks / elapsed * 100:.1f}%"
        self.previous_ticks = ticks
        self.previous_time = now
        gpu = self._gpu_memory(pids)
        return cpu, gpu

    @staticmethod
    def _gpu_memory(pids: set[int]) -> str:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=1,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return "—"
        used = 0
        for line in result.stdout.splitlines():
            try:
                pid, memory = (part.strip() for part in line.split(",", 1))
                if int(pid) in pids:
                    used += int(memory)
            except ValueError:
                continue
        return f"{used} MiB" if used else "0 MiB"


class JobWindow(QMainWindow):
    completed = Signal(object)

    def __init__(self, spec: JobSpec, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.spec = spec
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.started.connect(self._started)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._process_error)
        self.resources = ProcessResources()
        self.started_at: float | None = None
        self._last_chunk = ""
        mode = "Обучение" if spec.mode == "train" else "Тестирование"
        self.setWindowTitle(f"{mode} · {MODEL_NAMES[spec.model]}")
        self.resize(1080, 720)

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(20, 18, 20, 18)
        header = QHBoxLayout()
        title = QLabel(f"{mode}: {MODEL_NAMES[spec.model]}")
        title.setObjectName("sectionTitle")
        header.addWidget(title)
        header.addStretch()
        self.status = QLabel("Подготовка…")
        self.status.setObjectName("statusBadge")
        header.addWidget(self.status)
        root.addLayout(header)
        self.terminal = QPlainTextEdit()
        self.terminal.setObjectName("terminal")
        self.terminal.setReadOnly(True)
        self.terminal.setFont(QFont("JetBrains Mono", 10))
        root.addWidget(self.terminal, 1)

        metrics = QFrame()
        metrics.setObjectName("metricsBar")
        grid = QGridLayout(metrics)
        self.metric_values: dict[str, QLabel] = {}
        fields = (
            [("loss", "Loss"), ("eta", "Осталось"), ("gpu", "GPU"), ("cpu", "CPU")]
            if spec.mode == "train"
            else [("mae", "Action MAE"), ("rmse", "Action RMSE"), ("eta", "Осталось"), ("gpu", "GPU"), ("cpu", "CPU")]
        )
        for column, (key, label) in enumerate(fields):
            name = QLabel(label)
            name.setObjectName("metricName")
            value = QLabel("—")
            value.setObjectName("metricValue")
            value.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.metric_values[key] = value
            grid.addWidget(name, 0, column)
            grid.addWidget(value, 1, column)
        root.addWidget(metrics)
        buttons = QHBoxLayout()
        output_label = QLabel(f"Результаты: {spec.output_dir}")
        output_label.setObjectName("mutedLabel")
        buttons.addWidget(output_label)
        buttons.addStretch()
        self.stop_button = QPushButton("Остановить")
        self.stop_button.setObjectName("dangerButton")
        self.stop_button.clicked.connect(self.stop)
        buttons.addWidget(self.stop_button)
        root.addLayout(buttons)
        self.setCentralWidget(central)

        self.resource_timer = QTimer(self)
        self.resource_timer.setInterval(750)
        self.resource_timer.timeout.connect(self._update_resources)
        QTimer.singleShot(0, self.start)

    def start(self) -> None:
        output = Path(self.spec.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self._write_manifest("running")
        self.terminal.appendPlainText(f"$ {self.spec.program} {' '.join(self.spec.arguments)}\n")
        self.process.setWorkingDirectory(self.spec.working_directory)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        # Keep the installed Studio package importable when the process cwd is a submodule.
        source_root = str(Path(__file__).resolve().parents[2])
        current_pythonpath = environment.value("PYTHONPATH")
        environment.insert("PYTHONPATH", os.pathsep.join(filter(None, [source_root, current_pythonpath])))
        self.process.setProcessEnvironment(environment)
        self.process.start(self.spec.program, self.spec.arguments)

    def _started(self) -> None:
        self.started_at = time.monotonic()
        self.status.setText("Выполняется")
        self._update_resources()
        self.resource_timer.start()

    def _read_output(self) -> None:
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if not text:
            return
        self.terminal.moveCursor(QTextCursor.End)
        self.terminal.insertPlainText(text)
        self.terminal.moveCursor(QTextCursor.End)
        self._last_chunk = (self._last_chunk + text)[-8000:]
        self._parse_metrics(self._last_chunk)

    def _parse_metrics(self, text: str) -> None:
        patterns = {
            "loss": r"(?:\[METRIC\].*?loss=|['\"]?loss['\"]?\s*[:=]\s*)([0-9.eE+-]+)",
            "mae": r"(?:\[METRIC\].*?mae=|(?:Average\s+)?(?:Action\s+)?MAE[^0-9]*)([0-9.eE+-]+)",
            "rmse": r"(?:\[METRIC\].*?rmse=|(?:Average\s+)?(?:Action\s+)?RMSE[^0-9]*)([0-9.eE+-]+)",
            "eta": r"eta_seconds=([0-9.]+)",
        }
        for key, pattern in patterns.items():
            if key not in self.metric_values:
                continue
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                value = float(matches[-1])
                self.metric_values[key].setText(self._duration(value) if key == "eta" else f"{value:.5f}")
        mse = re.findall(r"Average MSE across all trajs:\s*([0-9.eE+-]+)", text, re.IGNORECASE)
        if mse and "rmse" in self.metric_values:
            self.metric_values["rmse"].setText(f"{math.sqrt(float(mse[-1])):.5f}")
        if "eta" in self.metric_values and "eta_seconds=" not in text and self.started_at:
            steps = re.findall(r"(?:step|steps?)\s*[=:]?\s*(\d+)\s*/\s*(\d+)", text, re.IGNORECASE)
            if steps:
                current, total = map(int, steps[-1])
                if current:
                    elapsed = time.monotonic() - self.started_at
                    self.metric_values["eta"].setText(self._duration(elapsed / current * (total - current)))

    @staticmethod
    def _duration(seconds: float) -> str:
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _update_resources(self) -> None:
        pid = int(self.process.processId())
        if not pid:
            return
        cpu, gpu = self.resources.sample(pid)
        self.metric_values["cpu"].setText(cpu)
        self.metric_values["gpu"].setText(gpu)

    def _finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._read_output()
        self.resource_timer.stop()
        succeeded = exit_code == 0
        self.status.setText("Завершено" if succeeded else f"Ошибка · код {exit_code}")
        self.stop_button.setText("Закрыть")
        self.stop_button.setObjectName("")
        self.stop_button.clicked.disconnect()
        self.stop_button.clicked.connect(self.close)
        self._write_manifest("completed" if succeeded else "failed", exit_code)
        self.completed.emit(self.spec)

    def _process_error(self, error: QProcess.ProcessError) -> None:
        self.terminal.appendPlainText(f"\n[Studio] Не удалось запустить процесс: {error.name}")

    def _write_manifest(self, status: str, exit_code: int | None = None) -> None:
        if self.spec.mode != "train":
            return
        payload = self.spec.public_payload()
        payload.update({"status": status, "exit_code": exit_code})
        manifest = Path(self.spec.output_dir) / "studio_checkpoint.json"
        manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def stop(self) -> None:
        if self.process.state() == QProcess.NotRunning:
            self.close()
            return
        self.status.setText("Остановка…")
        self.process.terminate()
        QTimer.singleShot(5000, self._kill_if_running)

    def _kill_if_running(self) -> None:
        if self.process.state() != QProcess.NotRunning:
            self.process.kill()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self.process.state() != QProcess.NotRunning:
            answer = QMessageBox.question(
                self,
                "Остановить процесс?",
                "Задание ещё выполняется. Остановить его и закрыть окно?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self.stop()
        event.accept()
