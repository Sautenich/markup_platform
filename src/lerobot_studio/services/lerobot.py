from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow.parquet as pq

from lerobot_studio.domain import DatasetInfo, Episode, VideoSegment


class DatasetError(RuntimeError):
    pass


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _first(row: dict[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None and not (isinstance(value, float) and pd.isna(value)):
            return value
    return default


class LeRobotDataset:
    """Version-tolerant reader plus a small, explicit Studio overlay.

    LeRobot v3 stores many episodes in a shared shard. Studio therefore keeps
    reversible per-episode edits and deletions in ``meta/lerobot_studio.json``.
    Generated episodes are real parquet files under ``studio/episodes`` and are
    described by ``studio/augmentations.json``.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        info_path = self.root / "meta" / "info.json"
        if not info_path.is_file():
            raise DatasetError(f"Не найден meta/info.json в {self.root}")
        self.raw_info: dict[str, Any] = _read_json(info_path, {})
        if not self.raw_info:
            raise DatasetError("meta/info.json повреждён или пуст")
        self.info = DatasetInfo(
            root=self.root,
            name=self.root.name,
            robot_type=str(self.raw_info.get("robot_type", "unknown")),
            fps=float(self.raw_info.get("fps", 30.0)),
            total_episodes=int(self.raw_info.get("total_episodes", 0)),
            total_frames=int(self.raw_info.get("total_frames", 0)),
            version=str(self.raw_info.get("codebase_version", "unknown")),
            features=dict(self.raw_info.get("features", {})),
        )
        self._episodes: list[Episode] | None = None

    @staticmethod
    def validate(path: Path | str) -> bool:
        return (Path(path).expanduser() / "meta" / "info.json").is_file()

    @property
    def studio_path(self) -> Path:
        return self.root / "meta" / "lerobot_studio.json"

    @property
    def augmentation_path(self) -> Path:
        return self.root / "studio" / "augmentations.json"

    def _studio(self) -> dict[str, Any]:
        return _read_json(self.studio_path, {"deleted": [], "edits": {}})

    def _tasks(self) -> dict[int, str]:
        result: dict[int, str] = {}
        parquet_path = self.root / "meta" / "tasks.parquet"
        jsonl_path = self.root / "meta" / "tasks.jsonl"
        try:
            if parquet_path.exists():
                frame = pd.read_parquet(parquet_path)
                for _, row in frame.iterrows():
                    result[int(row.get("task_index", len(result)))] = str(row.get("task", ""))
            elif jsonl_path.exists():
                for line in jsonl_path.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    result[int(row.get("task_index", len(result)))] = str(row.get("task", ""))
        except Exception:
            pass
        return result

    def episodes(self, refresh: bool = False) -> list[Episode]:
        if self._episodes is not None and not refresh:
            return self._episodes
        episodes = self._read_episode_index()
        studio = self._studio()
        deleted = {int(value) for value in studio.get("deleted", [])}
        edits = studio.get("edits", {})
        episodes.extend(self._read_augmented(episodes))
        for episode in episodes:
            edit = edits.get(str(episode.index), {})
            if "prompt" in edit:
                episode.prompt = str(edit["prompt"])
            episode.metadata.update(edit.get("metadata", {}))
        self._episodes = sorted((item for item in episodes if item.index not in deleted), key=lambda item: item.index)
        return self._episodes

    def _read_episode_index(self) -> list[Episode]:
        tasks = self._tasks()
        index_files = sorted((self.root / "meta" / "episodes").glob("**/*.parquet"))
        episodes: list[Episode] = []
        if index_files:
            for path in index_files:
                frame = pd.read_parquet(path)
                for row_data in frame.to_dict(orient="records"):
                    episodes.append(self._episode_from_row(row_data, tasks))
            return episodes

        # v2-style: one parquet file per episode.
        data_files = sorted((self.root / "data").glob("**/*.parquet"))
        for path in data_files:
            table = pq.read_table(path, columns=[name for name in ("episode_index", "task_index") if name in pq.read_schema(path).names])
            frame = table.to_pandas()
            if "episode_index" not in frame:
                continue
            for episode_index, group in frame.groupby("episode_index", sort=True):
                task_index = int(group["task_index"].iloc[0]) if "task_index" in group else 0
                index = int(episode_index)
                episodes.append(
                    Episode(
                        index=index,
                        length=len(group),
                        prompt=tasks.get(task_index, ""),
                        metadata={"fps": self.info.fps, "task_index": task_index},
                        videos=self._videos_for(index, {}),
                        data_file=path,
                    )
                )
        return episodes

    def _episode_from_row(self, row: dict[str, Any], tasks: dict[int, str]) -> Episode:
        index = int(_first(row, ("episode_index", "index"), 0))
        length = int(_first(row, ("length", "episode_length", "num_frames"), 0))
        task_values = row.get("tasks")
        if hasattr(task_values, "tolist"):
            task_values = task_values.tolist()
        prompt = ""
        if isinstance(task_values, (list, tuple)) and task_values:
            prompt = str(task_values[0])
        task_index = int(_first(row, ("task_index",), 0))
        if not prompt:
            prompt = tasks.get(task_index, "")
        chunk_index = int(_first(row, ("data/chunk_index", "data_chunk_index", "chunk_index"), index // int(self.raw_info.get("chunks_size", 1000))))
        file_index = int(_first(row, ("data/file_index", "data_file_index", "file_index"), index))
        data_path = self._format_path(self.raw_info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"), index, chunk_index, file_index)
        metadata = {key: self._json_value(value) for key, value in row.items()}
        metadata.setdefault("fps", self.info.fps)
        return Episode(
            index=index,
            length=length,
            prompt=prompt,
            metadata=metadata,
            videos=self._videos_for(index, row),
            data_file=data_path,
            data_start=self._as_int(_first(row, ("dataset_from_index", "data/from", "from_index"))),
            data_end=self._as_int(_first(row, ("dataset_to_index", "data/to", "to_index"))),
        )

    @staticmethod
    def _json_value(value: Any) -> Any:
        if hasattr(value, "tolist"):
            return value.tolist()
        if hasattr(value, "item"):
            try:
                return value.item()
            except ValueError:
                pass
        return value

    @staticmethod
    def _as_int(value: Any) -> int | None:
        return None if value is None else int(value)

    def _format_path(self, pattern: str, episode: int, chunk: int, file_index: int, video_key: str = "") -> Path:
        values = {
            "episode_index": episode,
            "episode_chunk": episode // int(self.raw_info.get("chunks_size", 1000)),
            "chunk_index": chunk,
            "file_index": file_index,
            "video_key": video_key,
        }
        try:
            return self.root / pattern.format(**values)
        except (KeyError, ValueError):
            return self.root / pattern

    def _videos_for(self, index: int, row: dict[str, Any]) -> list[VideoSegment]:
        segments: list[VideoSegment] = []
        pattern = self.raw_info.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
        for key in self.info.video_keys:
            prefixes = (f"videos/{key}", f"video/{key}", key)
            chunk = int(_first(row, tuple(f"{prefix}/chunk_index" for prefix in prefixes), index // int(self.raw_info.get("chunks_size", 1000))))
            file_index = int(_first(row, tuple(f"{prefix}/file_index" for prefix in prefixes), index))
            start = float(_first(row, tuple(f"{prefix}/from_timestamp" for prefix in prefixes) + tuple(f"{prefix}/start_timestamp" for prefix in prefixes), 0.0))
            end_value = _first(row, tuple(f"{prefix}/to_timestamp" for prefix in prefixes) + tuple(f"{prefix}/end_timestamp" for prefix in prefixes))
            feature = self.info.features.get(key, {})
            has_audio = bool(feature.get("info", {}).get("has_audio", False))
            segments.append(VideoSegment(key, self._format_path(pattern, index, chunk, file_index, key), start, float(end_value) if end_value is not None else None, has_audio))
        return segments

    def _read_augmented(self, originals: list[Episode]) -> list[Episode]:
        by_index = {item.index: item for item in originals}
        rows = _read_json(self.augmentation_path, [])
        result: list[Episode] = []
        for row in rows:
            source = by_index.get(int(row.get("source_index", -1)))
            data_file = self.root / row["data_file"]
            if source is None or not data_file.exists():
                continue
            augmentation_type = row.get("augmentation")
            augmentation_metadata = {
                **source.metadata,
                **row.get("metadata", {}),
                "is_augmentation": True,
                "augmentation": augmentation_type,
                "augmentation_type": augmentation_type,
                "source_episode_index": source.index,
            }
            videos = source.videos
            if row.get("videos"):
                videos = [
                    VideoSegment(
                        key=str(item["key"]),
                        path=self.root / item["path"],
                        start_s=float(item.get("start_s", 0.0)),
                        end_s=float(item["end_s"]) if item.get("end_s") is not None else None,
                        has_audio=bool(item.get("has_audio", False)),
                    )
                    for item in row["videos"]
                ]
            result.append(
                Episode(
                    index=int(row["episode_index"]),
                    length=int(row.get("length", source.length)),
                    prompt=str(row.get("prompt", source.prompt)),
                    metadata=augmentation_metadata,
                    videos=videos,
                    data_file=data_file,
                    augmented=True,
                    source_index=source.index,
                )
            )
        return result

    def dataframe(self, episode: Episode) -> pd.DataFrame:
        if episode.data_file is None or not episode.data_file.exists():
            raise DatasetError(f"Parquet эпизода {episode.index} ещё не загружен")
        frame = pd.read_parquet(episode.data_file)
        if "episode_index" in frame.columns and not episode.augmented:
            frame = frame[frame["episode_index"] == episode.index]
        if frame.empty:
            raise DatasetError(f"В parquet не найдены кадры эпизода {episode.index}")
        return frame.reset_index(drop=True)

    def edit_episode(self, episode_index: int, prompt: str, metadata: dict[str, Any]) -> None:
        studio = self._studio()
        studio.setdefault("edits", {})[str(episode_index)] = {"prompt": prompt, "metadata": metadata}
        _write_json(self.studio_path, studio)
        self._episodes = None

    def soft_delete(self, episode: Episode, trash_root: Path) -> Path:
        destination = trash_root.expanduser().resolve() / self.info.name / f"episode_{episode.index:06d}"
        if destination.exists():
            raise DatasetError(f"Папка уже существует: {destination}")
        destination.mkdir(parents=True)
        try:
            frame = self.dataframe(episode)
            frame.to_parquet(destination / "episode.parquet", index=False, compression="zstd")
            video_meta: list[dict[str, Any]] = []
            for segment in episode.videos:
                if not segment.path.exists():
                    raise DatasetError(f"Видео ещё не загружено: {segment.path}")
                target = destination / f"{segment.key.replace('/', '_')}.mp4"
                self._extract_video(segment, target, episode.duration)
                video_meta.append({"key": segment.key, "file": target.name})
            _write_json(destination / "metadata.json", {
                "source_dataset": str(self.root),
                "episode_index": episode.index,
                "prompt": episode.prompt,
                "metadata": episode.metadata,
                "videos": video_meta,
            })
        except Exception:
            shutil.rmtree(destination, ignore_errors=True)
            raise
        studio = self._studio()
        studio["deleted"] = sorted(set(int(value) for value in studio.get("deleted", [])) | {episode.index})
        _write_json(self.studio_path, studio)
        self._episodes = None
        return destination

    @staticmethod
    def _extract_video(segment: VideoSegment, target: Path, fallback_duration: float) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        duration = (segment.end_s - segment.start_s) if segment.end_s is not None else fallback_duration
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{segment.start_s:.6f}", "-i", str(segment.path),
            "-t", f"{max(duration, 0.001):.6f}", "-map", "0", "-c", "copy", str(target),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True)
        except FileNotFoundError as error:
            raise DatasetError("Для экспорта видео требуется ffmpeg") from error
        except subprocess.CalledProcessError as error:
            detail = error.stderr.decode(errors="replace").strip()
            raise DatasetError(f"ffmpeg не смог извлечь эпизод: {detail}") from error
