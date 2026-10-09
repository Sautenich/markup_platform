from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable

import numpy as np

from lerobot_studio.domain import Episode
from lerobot_studio.services.lerobot import LeRobotDataset, _write_json


def _frame_statistics(frame, features: dict) -> dict:
    statistics: dict = {}
    for key, feature in features.items():
        if key not in frame or feature.get("dtype") == "video":
            continue
        try:
            first = frame[key].iloc[0]
            values = np.stack(frame[key].to_numpy()) if np.ndim(first) else frame[key].to_numpy()[:, None]
            if not np.issubdtype(values.dtype, np.number):
                continue
            values = values.astype(np.float64)
        except (TypeError, ValueError):
            continue
        statistics[key] = {
            "min": np.min(values, axis=0).tolist(),
            "max": np.max(values, axis=0).tolist(),
            "mean": np.mean(values, axis=0).tolist(),
            "std": np.std(values, axis=0).tolist(),
            "count": [len(values)],
        }
    return statistics


def _combine_statistics(rows: list[dict]) -> dict:
    combined: dict = {}
    keys = {key for row in rows for key in row}
    for key in keys:
        entries = [row[key] for row in rows if key in row]
        counts = np.array([entry["count"][0] for entry in entries], dtype=np.float64)
        total = float(np.sum(counts))
        means = np.stack([entry["mean"] for entry in entries]).astype(np.float64)
        stds = np.stack([entry["std"] for entry in entries]).astype(np.float64)
        mean = np.sum(means * counts[:, None], axis=0) / total
        second_moment = np.sum((stds**2 + means**2) * counts[:, None], axis=0) / total
        combined[key] = {
            "min": np.min(np.stack([entry["min"] for entry in entries]), axis=0).tolist(),
            "max": np.max(np.stack([entry["max"] for entry in entries]), axis=0).tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(np.maximum(0.0, second_moment - mean**2)).tolist(),
            "count": [int(total)],
        }
    return combined


def write_dataset_statistics(dataset: LeRobotDataset, image_stats_source: Path | None = None) -> None:
    """Recalculate tabular statistics and write v2-style metadata files."""
    rows: list[dict] = []
    episode_rows: list[dict] = []
    for episode in dataset.episodes():
        stats = _frame_statistics(dataset.dataframe(episode), dataset.info.features)
        rows.append(stats)
        episode_rows.append({"episode_index": episode.index, "stats": stats})
    combined = _combine_statistics(rows)
    if image_stats_source and image_stats_source.exists():
        source_stats = json.loads(image_stats_source.read_text(encoding="utf-8"))
        for key in dataset.info.video_keys:
            if key in source_stats:
                combined[key] = source_stats[key]
    _write_json(dataset.root / "meta" / "stats.json", combined)
    (dataset.root / "meta" / "episodes_stats.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in episode_rows) + "\n",
        encoding="utf-8",
    )


class SplitExporter:
    def __init__(self, dataset: LeRobotDataset) -> None:
        self.dataset = dataset

    def partition(self, train: int, validation: int, test: int, seed: int = 42) -> dict[str, list[Episode]]:
        if train + validation + test != 100 or min(train, validation, test) < 0:
            raise ValueError("Доли должны быть неотрицательными и в сумме давать 100%")
        episodes = self.dataset.episodes()
        order = np.random.default_rng(seed).permutation(len(episodes))
        train_end = round(len(episodes) * train / 100)
        val_end = train_end + round(len(episodes) * validation / 100)
        return {
            "train": [episodes[int(i)] for i in order[:train_end]],
            "validation": [episodes[int(i)] for i in order[train_end:val_end]],
            "test": [episodes[int(i)] for i in order[val_end:]],
        }

    def export(
        self,
        output_parent: Path,
        ratios: tuple[int, int, int],
        progress: Callable[[str, int, int], None] | None = None,
    ) -> list[Path]:
        partitions = self.partition(*ratios)
        created: list[Path] = []
        for split_name, episodes in partitions.items():
            if not episodes:
                continue
            target = output_parent / f"{self.dataset.info.name}_{split_name}"
            if target.exists() and any(target.iterdir()):
                raise FileExistsError(f"Целевая папка не пуста: {target}")
            target.mkdir(parents=True, exist_ok=True)
            try:
                self._export_one(target, split_name, episodes, progress)
            except Exception:
                shutil.rmtree(target, ignore_errors=True)
                raise
            created.append(target)
        return created

    def export_selection(
        self,
        target: Path,
        episodes: list[Episode],
        name: str = "train",
        progress: Callable[[str, int, int], None] | None = None,
    ) -> Path:
        """Materialize an explicit episode selection as a standalone dataset."""
        if not episodes:
            raise ValueError("Для экспорта не выбрано ни одного эпизода")
        if target.exists() and any(target.iterdir()):
            raise FileExistsError(f"Целевая папка не пуста: {target}")
        target.mkdir(parents=True, exist_ok=True)
        try:
            self._export_one(target, name, episodes, progress)
        except Exception:
            shutil.rmtree(target, ignore_errors=True)
            raise
        return target

    def _export_one(
        self,
        target: Path,
        split_name: str,
        episodes: list[Episode],
        progress: Callable[[str, int, int], None] | None,
    ) -> None:
        info = dict(self.dataset.raw_info)
        info.update({
            "codebase_version": "v2.1",
            "total_episodes": len(episodes),
            "total_frames": sum(item.length for item in episodes),
            "total_tasks": len({item.prompt for item in episodes}),
            "total_videos": len(episodes) * len(self.dataset.info.video_keys),
            "total_chunks": max(1, (len(episodes) + 999) // 1000),
            "chunks_size": 1000,
            "splits": {split_name: f"0:{len(episodes)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        })
        tasks: dict[str, int] = {}
        episode_rows: list[dict] = []
        global_index = 0
        for new_index, source in enumerate(episodes):
            frame = self.dataset.dataframe(source).copy()
            prompt = source.prompt
            task_index = tasks.setdefault(prompt, len(tasks))
            frame["episode_index"] = new_index
            if "frame_index" in frame:
                frame["frame_index"] = np.arange(len(frame), dtype=np.int64)
            if "index" in frame:
                frame["index"] = np.arange(global_index, global_index + len(frame), dtype=np.int64)
            if "task_index" in frame:
                frame["task_index"] = task_index
            chunk = new_index // 1000
            data_target = target / "data" / f"chunk-{chunk:03d}" / f"episode_{new_index:06d}.parquet"
            data_target.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(data_target, index=False, compression="zstd")
            for segment in source.videos:
                if not segment.path.exists():
                    raise FileNotFoundError(f"Видео ещё не загружено: {segment.path}")
                video_target = target / "videos" / f"chunk-{chunk:03d}" / segment.key / f"episode_{new_index:06d}.mp4"
                LeRobotDataset._extract_video(segment, video_target, source.duration)
            episode_rows.append({"episode_index": new_index, "tasks": [prompt], "length": len(frame)})
            global_index += len(frame)
            if progress:
                progress(split_name, new_index + 1, len(episodes))
        _write_json(target / "meta" / "info.json", info)
        (target / "meta").mkdir(parents=True, exist_ok=True)
        (target / "meta" / "episodes.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in episode_rows) + "\n", encoding="utf-8"
        )
        task_rows = [{"task_index": value, "task": key} for key, value in tasks.items()]
        (target / "meta" / "tasks.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in task_rows) + "\n", encoding="utf-8"
        )
        write_dataset_statistics(
            LeRobotDataset(target),
            self.dataset.root / "meta" / "stats.json",
        )
        readme = self.dataset.root / "README.md"
        if readme.exists():
            shutil.copy2(readme, target / "README.md")
