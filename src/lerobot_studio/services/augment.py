from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from lerobot_studio.domain import Episode
from lerobot_studio.services.lerobot import LeRobotDataset


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AugmentationSpec:
    key: str
    title: str
    description: str


AUGMENTATIONS = [
    AugmentationSpec("joint_noise", "Шум суставов", "Добавляет небольшой гауссов шум к состояниям и действиям."),
    AugmentationSpec("trajectory", "Вариация траектории", "Плавно отклоняет траекторию, сохраняя начало и конец."),
    AugmentationSpec("time_warp", "Темп движения", "Нелинейно ускоряет и замедляет отдельные участки."),
    AugmentationSpec("sensor_drift", "Дрейф сенсоров", "Добавляет медленно меняющееся смещение сенсоров."),
    AugmentationSpec("short_dropout", "Короткие пропуски", "Имитирует пропуски измерений с интерполяцией."),
]


@dataclass(frozen=True, slots=True)
class VisualAugmentationSpec:
    key: str
    title: str
    description: str
    minimum: float
    maximum: float
    default: float
    step: float
    unit: str = ""


VISUAL_AUGMENTATIONS = [
    VisualAugmentationSpec("brightness", "Яркость", "Осветляет или затемняет изображение.", -0.5, 0.5, 0.10, 0.01),
    VisualAugmentationSpec("contrast", "Контраст", "Изменяет различие между светлыми и тёмными областями.", 0.5, 2.0, 1.20, 0.05, "×"),
    VisualAugmentationSpec("saturation", "Насыщенность", "Ослабляет или усиливает цвета.", 0.0, 3.0, 1.30, 0.05, "×"),
    VisualAugmentationSpec("hue", "Сдвиг оттенка", "Сдвигает цветовой тон по цветовому кругу.", -180.0, 180.0, 12.0, 1.0, "°"),
    VisualAugmentationSpec("gamma", "Гамма", "Меняет яркость средних тонов.", 0.5, 2.0, 1.10, 0.05, "×"),
    VisualAugmentationSpec("blur", "Размытие", "Имитирует дефокусировку или движение камеры.", 0.1, 8.0, 1.50, 0.10, "σ"),
    VisualAugmentationSpec("noise", "Шум камеры", "Добавляет яркостный шум сенсора.", 1.0, 50.0, 8.0, 1.0),
    VisualAugmentationSpec("vignette", "Виньетирование", "Затемняет края кадра.", 0.05, 1.20, 0.45, 0.05, "rad"),
    VisualAugmentationSpec("zoom", "Кроп / приближение", "Обрезает края и возвращает исходный размер кадра.", 1.01, 1.50, 1.10, 0.01, "×"),
    VisualAugmentationSpec("rotation", "Поворот", "Немного поворачивает кадр с чёрным заполнением краёв.", -15.0, 15.0, 3.0, 0.5, "°"),
]


class AugmentationService:
    def __init__(self, dataset: LeRobotDataset, seed: int | None = None) -> None:
        self.dataset = dataset
        self.rng = np.random.default_rng(seed)

    def create(
        self,
        augmentation: str,
        count: int,
        progress: Callable[[int, int], None] | None = None,
    ) -> list[int]:
        if augmentation not in {item.key for item in AUGMENTATIONS}:
            raise ValueError(f"Неизвестная аугментация: {augmentation}")
        if count < 1:
            raise ValueError("Количество должно быть положительным")
        candidates = [item for item in self.dataset.episodes() if not item.augmented]
        if not candidates:
            raise ValueError("В датасете нет исходных эпизодов")
        sources = self._select_sources(candidates, count)
        existing = json.loads(self.dataset.augmentation_path.read_text(encoding="utf-8")) if self.dataset.augmentation_path.exists() else []
        next_index = self._next_episode_index()
        output_dir = self.dataset.root / "studio" / "episodes"
        output_dir.mkdir(parents=True, exist_ok=True)
        operation_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        journal = self.dataset.root / "studio" / "operations" / f"signals-{operation_id}.json"
        operation: dict = {
            "operation_id": operation_id,
            "kind": "signal_augmentation",
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "dataset": str(self.dataset.root),
            "augmentation": augmentation,
            "requested_count": count,
            "sources": [source.index for source in sources],
            "next_episode_index": next_index,
            "created": [],
        }
        self._write_json(journal, operation)
        self._dataset_log(
            "START SIGNAL operation=%s kind=%s count=%s next_index=%s sources=%s",
            operation_id, augmentation, count, next_index, operation["sources"],
        )
        created: list[int] = []
        for position, source in enumerate(sources, start=1):
            episode_index = next_index + position - 1
            target = output_dir / f"episode_{episode_index:06d}.parquet"
            operation.update({
                "current_position": position,
                "current_source": source.index,
                "current_episode": episode_index,
            })
            self._write_json(journal, operation)
            try:
                frame = self.dataset.dataframe(source).copy()
                transformed, parameters = self._transform(frame, augmentation)
                if "episode_index" in transformed:
                    transformed["episode_index"] = episode_index
                if "frame_index" in transformed:
                    transformed["frame_index"] = np.arange(len(transformed), dtype=np.int64)
                transformed.to_parquet(target, index=False, compression="zstd")
            except Exception as error:
                target.unlink(missing_ok=True)
                operation.update({
                    "status": "failed",
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                    "error": f"{type(error).__name__}: {error}",
                })
                self._write_json(journal, operation)
                self._dataset_log(
                    "FAILED SIGNAL operation=%s episode=%s error=%r",
                    operation_id, episode_index, error, level=logging.ERROR,
                )
                raise
            existing.append({
                "episode_index": episode_index,
                "source_index": source.index,
                "data_file": str(target.relative_to(self.dataset.root)),
                "length": len(transformed),
                "prompt": source.prompt,
                "augmentation": augmentation,
                "metadata": {
                    "is_augmentation": True,
                    "augmentation_type": augmentation,
                    "augmentation_parameters": parameters,
                    "source_episode_index": source.index,
                    "fps": self.dataset.info.fps,
                },
            })
            self._write_json(self.dataset.augmentation_path, existing)
            created.append(episode_index)
            operation["created"] = list(created)
            self._write_json(journal, operation)
            self._dataset_log("COMMIT SIGNAL operation=%s episode=%s", operation_id, episode_index)
            if progress:
                progress(position, count)
        operation.update({
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
        self._write_json(journal, operation)
        self._dataset_log("COMPLETE SIGNAL operation=%s created=%s", operation_id, created)
        self.dataset.episodes(refresh=True)
        return created

    def create_visual(
        self,
        parameters: dict[str, float],
        count: int,
        progress: Callable[[int, int], None] | None = None,
    ) -> list[int]:
        allowed = {item.key for item in VISUAL_AUGMENTATIONS}
        if not parameters or not set(parameters).issubset(allowed):
            raise ValueError("Выберите хотя бы одну визуальную аугментацию")
        if count < 1:
            raise ValueError("Количество должно быть положительным")
        candidates = [item for item in self.dataset.episodes() if not item.augmented]
        if not candidates:
            raise ValueError("В датасете нет исходных эпизодов")
        if not self.dataset.info.video_keys:
            raise ValueError("В датасете нет видео для визуальной аугментации")
        sources = self._select_sources(candidates, count)
        existing = json.loads(self.dataset.augmentation_path.read_text(encoding="utf-8")) if self.dataset.augmentation_path.exists() else []
        next_index = self._next_episode_index()
        data_dir = self.dataset.root / "studio" / "episodes"
        video_root = self.dataset.root / "studio" / "videos"
        data_dir.mkdir(parents=True, exist_ok=True)
        video_root.mkdir(parents=True, exist_ok=True)
        operation_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        journal = self.dataset.root / "studio" / "operations" / f"visual-{operation_id}.json"
        operation: dict = {
            "operation_id": operation_id,
            "kind": "visual_augmentation",
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "dataset": str(self.dataset.root),
            "parameters": parameters,
            "requested_count": count,
            "sources": [source.index for source in sources],
            "next_episode_index": next_index,
            "created": [],
        }
        self._write_json(journal, operation)
        self._dataset_log(
            "START operation=%s count=%s next_index=%s sources=%s parameters=%s",
            operation_id, count, next_index, operation["sources"], parameters,
        )
        created: list[int] = []
        for position, source in enumerate(sources, start=1):
            episode_index = next_index + position - 1
            data_target = data_dir / f"episode_{episode_index:06d}.parquet"
            video_dir = video_root / f"episode_{episode_index:06d}"
            frame = self.dataset.dataframe(source).copy()
            if "episode_index" in frame:
                frame["episode_index"] = episode_index
            if "frame_index" in frame:
                frame["frame_index"] = np.arange(len(frame), dtype=np.int64)
            video_rows: list[dict] = []
            try:
                operation.update({
                    "current_position": position,
                    "current_source": source.index,
                    "current_episode": episode_index,
                    "current_camera": None,
                })
                self._write_json(journal, operation)
                self._dataset_log(
                    "EPISODE operation=%s position=%s/%s source=%s target=%s",
                    operation_id, position, count, source.index, episode_index,
                )
                frame.to_parquet(data_target, index=False, compression="zstd")
                for segment in source.videos:
                    if not segment.path.exists():
                        raise FileNotFoundError(f"Видео ещё не загружено: {segment.path}")
                    target = video_dir / f"{segment.key.replace('/', '__')}.mp4"
                    feature = self.dataset.info.features.get(segment.key, {})
                    info = feature.get("info", {})
                    width = int(info.get("video.width", 640))
                    height = int(info.get("video.height", 480))
                    operation["current_camera"] = segment.key
                    operation["current_output"] = str(target)
                    self._write_json(journal, operation)
                    self._dataset_log(
                        "CAMERA operation=%s episode=%s key=%s input=%s output=%s",
                        operation_id, episode_index, segment.key, segment.path, target,
                    )
                    self._render_video(segment, target, source.duration, parameters, width, height)
                    duration = (segment.end_s - segment.start_s) if segment.end_s is not None else source.duration
                    video_rows.append({
                        "key": segment.key,
                        "path": str(target.relative_to(self.dataset.root)),
                        "start_s": 0.0,
                        "end_s": duration,
                        "has_audio": segment.has_audio,
                    })
            except Exception as error:
                data_target.unlink(missing_ok=True)
                shutil.rmtree(video_dir, ignore_errors=True)
                operation.update({
                    "status": "failed",
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                    "error": f"{type(error).__name__}: {error}",
                })
                self._write_json(journal, operation)
                self._dataset_log(
                    "FAILED operation=%s episode=%s error=%r", operation_id, episode_index, error,
                    level=logging.ERROR,
                )
                LOGGER.exception("Visual augmentation failed operation=%s", operation_id)
                raise
            row = {
                "episode_index": episode_index,
                "source_index": source.index,
                "data_file": str(data_target.relative_to(self.dataset.root)),
                "length": len(frame),
                "prompt": source.prompt,
                "augmentation": "visual",
                "videos": video_rows,
                "metadata": {
                    "is_augmentation": True,
                    "augmentation_type": "visual",
                    "visual_augmentation_parameters": parameters,
                    "source_episode_index": source.index,
                    "video_codec": "h264",
                    "fps": self.dataset.info.fps,
                },
            }
            existing.append(row)
            # Commit every completed episode. A crash during a later render must
            # never leave already finished videos invisible to the application.
            self._write_json(self.dataset.augmentation_path, existing)
            created.append(episode_index)
            operation["created"] = list(created)
            operation["current_camera"] = None
            operation.pop("current_output", None)
            self._write_json(journal, operation)
            self._dataset_log("COMMIT operation=%s episode=%s", operation_id, episode_index)
            if progress:
                progress(position, count)
        operation.update({
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "current_camera": None,
        })
        self._write_json(journal, operation)
        self._dataset_log("COMPLETE operation=%s created=%s", operation_id, created)
        self.dataset.episodes(refresh=True)
        return created

    def _next_episode_index(self) -> int:
        """Reserve an index after both registered and orphaned Studio outputs."""
        indices = {item.index for item in self.dataset.episodes()}
        for path in (self.dataset.root / "studio" / "episodes").glob("episode_*.parquet"):
            try:
                indices.add(int(path.stem.removeprefix("episode_")))
            except ValueError:
                continue
        for path in (self.dataset.root / "studio" / "videos").glob("episode_*"):
            try:
                indices.add(int(path.name.removeprefix("episode_")))
            except ValueError:
                continue
        return max(indices, default=-1) + 1

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)

    def _dataset_log(self, message: str, *args: object, level: int = logging.INFO) -> None:
        rendered = message % args
        path = self.dataset.root / "studio" / "debug.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).isoformat()
        with path.open("a", encoding="utf-8") as stream:
            stream.write(f"{timestamp} {logging.getLevelName(level)} {rendered}\n")
        LOGGER.log(level, rendered)

    @staticmethod
    def video_filters(parameters: dict[str, float], width: int, height: int) -> str:
        filters: list[str] = []
        eq_values = {key: parameters[key] for key in ("brightness", "contrast", "saturation", "gamma") if key in parameters}
        if eq_values:
            filters.append("eq=" + ":".join(f"{key}={value:.4f}" for key, value in eq_values.items()))
        if "hue" in parameters:
            filters.append(f"hue=h={parameters['hue']:.3f}")
        if "blur" in parameters:
            filters.append(f"gblur=sigma={parameters['blur']:.3f}")
        if "noise" in parameters:
            filters.append(f"noise=alls={parameters['noise']:.3f}:allf=t+u")
        if "vignette" in parameters:
            filters.append(f"vignette=angle={parameters['vignette']:.4f}")
        if "zoom" in parameters:
            zoom = parameters["zoom"]
            filters.append(f"crop=trunc(iw/{zoom:.4f}/2)*2:trunc(ih/{zoom:.4f}/2)*2,scale={width}:{height}")
        if "rotation" in parameters:
            radians = parameters["rotation"] * math.pi / 180.0
            filters.append(f"rotate={radians:.7f}:ow=iw:oh=ih:fillcolor=black")
        return ",".join(filters)

    @classmethod
    def _render_video(
        cls,
        segment,
        target: Path,
        fallback_duration: float,
        parameters: dict[str, float],
        width: int,
        height: int,
    ) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        duration = (segment.end_s - segment.start_s) if segment.end_s is not None else fallback_duration
        filters = cls.video_filters(parameters, width, height)
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{segment.start_s:.6f}", "-i", str(segment.path),
            "-t", f"{max(duration, 0.001):.6f}", "-vf", filters,
            "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264",
            "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-movflags", "+faststart", str(target),
        ]
        LOGGER.info("Running ffmpeg: %s", subprocess.list2cmdline(command))
        started = time.monotonic()
        try:
            result = subprocess.run(command, check=True, capture_output=True)
            LOGGER.info(
                "ffmpeg completed in %.2fs output=%s bytes=%s",
                time.monotonic() - started,
                target,
                target.stat().st_size if target.exists() else 0,
            )
        except FileNotFoundError as error:
            raise RuntimeError("Для визуальной аугментации требуется ffmpeg") from error
        except subprocess.CalledProcessError as error:
            detail = error.stderr.decode(errors="replace").strip()
            raise RuntimeError(f"ffmpeg не смог применить визуальную аугментацию: {detail}") from error

    def _select_sources(self, episodes: list[Episode], count: int) -> list[Episode]:
        order = list(self.rng.permutation(len(episodes)))
        if count <= len(episodes):
            return [episodes[index] for index in order[:count]]
        selected = [episodes[index] for index in order]
        extra = self.rng.integers(0, len(episodes), size=count - len(episodes))
        selected.extend(episodes[int(index)] for index in extra)
        return selected

    def _numeric_vectors(self, frame: pd.DataFrame) -> list[str]:
        keys: list[str] = []
        for key, feature in self.dataset.info.features.items():
            shape = feature.get("shape", [])
            if key in frame and feature.get("dtype", "").startswith("float") and shape and int(np.prod(shape)) > 1:
                keys.append(key)
        return keys

    def _transform(self, frame: pd.DataFrame, kind: str) -> tuple[pd.DataFrame, dict[str, float | int]]:
        vector_keys = self._numeric_vectors(frame)
        if not vector_keys:
            raise ValueError("Нет векторных float-полей для аугментации")
        if kind == "joint_noise":
            sigma = float(self.rng.uniform(0.003, 0.015))
            for key in vector_keys:
                values = np.stack(frame[key].to_numpy())
                frame[key] = list((values + self.rng.normal(0.0, sigma, values.shape)).astype(values.dtype))
            return frame, {"sigma": sigma}
        if kind == "trajectory":
            amplitude = float(self.rng.uniform(0.01, 0.05))
            phase = np.sin(np.linspace(0, np.pi, len(frame)))[:, None]
            for key in vector_keys:
                values = np.stack(frame[key].to_numpy())
                direction = self.rng.normal(size=(1, values.shape[1]))
                direction /= np.linalg.norm(direction) + 1e-9
                frame[key] = list((values + amplitude * phase * direction).astype(values.dtype))
            return frame, {"amplitude": amplitude}
        if kind == "time_warp":
            exponent = float(self.rng.uniform(0.8, 1.25))
            source_positions = np.linspace(0, len(frame) - 1, len(frame)) ** exponent
            source_positions *= (len(frame) - 1) / max(source_positions[-1], 1)
            grid = np.arange(len(frame))
            for key in vector_keys:
                values = np.stack(frame[key].to_numpy())
                warped = np.column_stack([np.interp(source_positions, grid, values[:, dim]) for dim in range(values.shape[1])])
                frame[key] = list(warped.astype(values.dtype))
            return frame, {"exponent": exponent}
        if kind == "sensor_drift":
            maximum = float(self.rng.uniform(0.005, 0.03))
            drift = np.linspace(0.0, maximum, len(frame))[:, None]
            for key in vector_keys:
                values = np.stack(frame[key].to_numpy())
                signs = self.rng.choice((-1.0, 1.0), size=(1, values.shape[1]))
                frame[key] = list((values + drift * signs).astype(values.dtype))
            return frame, {"maximum": maximum}
        width = max(1, min(len(frame) // 20, int(self.rng.integers(2, 9))))
        start = int(self.rng.integers(1, max(2, len(frame) - width)))
        for key in vector_keys:
            values = np.stack(frame[key].to_numpy())
            left = values[max(0, start - 1)]
            right = values[min(len(values) - 1, start + width)]
            values[start : start + width] = np.linspace(left, right, width)
            frame[key] = list(values)
        return frame, {"start_frame": start, "frames": width}
