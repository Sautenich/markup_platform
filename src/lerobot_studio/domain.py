from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class VideoSegment:
    key: str
    path: Path
    start_s: float = 0.0
    end_s: float | None = None
    has_audio: bool = False


@dataclass(slots=True)
class Episode:
    index: int
    length: int
    prompt: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    videos: list[VideoSegment] = field(default_factory=list)
    data_file: Path | None = None
    data_start: int | None = None
    data_end: int | None = None
    augmented: bool = False
    source_index: int | None = None

    @property
    def duration(self) -> float:
        fps = float(self.metadata.get("fps", 30.0))
        return self.length / fps if fps else 0.0


@dataclass(slots=True)
class DatasetInfo:
    root: Path
    name: str
    robot_type: str
    fps: float
    total_episodes: int
    total_frames: int
    version: str
    features: dict[str, Any]

    @property
    def video_keys(self) -> list[str]:
        return [key for key, value in self.features.items() if value.get("dtype") == "video"]

    @property
    def imu_keys(self) -> list[str]:
        needles = ("imu", "gyro", "acceler", "orientation")
        return [key for key in self.features if any(item in key.lower() for item in needles)]

