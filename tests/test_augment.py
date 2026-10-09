from __future__ import annotations

import json
import subprocess

import numpy as np
import pandas as pd
import pytest

from lerobot_studio.services.augment import AUGMENTATIONS, VISUAL_AUGMENTATIONS, AugmentationService
from lerobot_studio.services.lerobot import LeRobotDataset


def make_dataset(root, episodes: int = 3) -> LeRobotDataset:
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "testbot",
        "total_episodes": episodes,
        "total_frames": episodes * 5,
        "fps": 10,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [2], "names": [["a", "b"]]},
            "action": {"dtype": "float32", "shape": [2], "names": [["a", "b"]]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    for index in range(episodes):
        values = [np.array([index + i, index - i], dtype=np.float32) for i in range(5)]
        pd.DataFrame({
            "observation.state": values,
            "action": values,
            "timestamp": np.arange(5) / 10,
            "episode_index": index,
            "frame_index": np.arange(5),
        }).to_parquet(root / "data" / "chunk-000" / f"episode_{index:06d}.parquet")
    return LeRobotDataset(root)


def test_every_augmentation_creates_editable_episode(tmp_path):
    dataset = make_dataset(tmp_path / "dataset")
    for spec in AUGMENTATIONS:
        created = AugmentationService(dataset, seed=1).create(spec.key, 1)
        assert len(created) == 1
        generated = next(item for item in dataset.episodes(refresh=True) if item.index == created[0])
        assert generated.augmented
        assert generated.metadata["is_augmentation"] is True
        assert generated.metadata["source_episode_index"] == generated.source_index
        assert len(dataset.dataframe(generated)) == 5
        dataset.edit_episode(generated.index, "edited augmented prompt", {"reviewed": True})
        edited = next(item for item in dataset.episodes(refresh=True) if item.index == generated.index)
        assert edited.prompt == "edited augmented prompt"
        assert edited.metadata["reviewed"] is True


def test_source_coverage_before_repetition(tmp_path):
    dataset = make_dataset(tmp_path / "dataset", episodes=4)
    AugmentationService(dataset, seed=4).create("joint_noise", 4)
    rows = json.loads(dataset.augmentation_path.read_text(encoding="utf-8"))
    assert len({row["source_index"] for row in rows}) == 4


def test_visual_augmentation_renders_new_video(tmp_path):
    root = tmp_path / "dataset"
    make_dataset(root, episodes=1)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"]["observation.images.camera"] = {
        "dtype": "video",
        "shape": [3, 48, 64],
        "info": {"video.width": 64, "video.height": 48, "has_audio": False},
    }
    info_path.write_text(json.dumps(info), encoding="utf-8")
    video = root / "videos" / "chunk-000" / "observation.images.camera" / "episode_000000.mp4"
    video.parent.mkdir(parents=True)
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10:duration=0.5",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video),
        ],
        check=True,
    )
    dataset = LeRobotDataset(root)
    visual_parameters = {spec.key: spec.default for spec in VISUAL_AUGMENTATIONS}
    created = AugmentationService(dataset, seed=2).create_visual(visual_parameters, 1)
    generated = next(item for item in dataset.episodes(refresh=True) if item.index == created[0])
    assert generated.augmented
    assert generated.metadata["augmentation_type"] == "visual"
    assert generated.metadata["visual_augmentation_parameters"]["brightness"] == 0.10
    assert len(generated.videos) == 1
    assert generated.videos[0].path.exists()
    assert generated.videos[0].path != video


def test_ten_visual_augmentations_have_ffmpeg_filters():
    assert len(VISUAL_AUGMENTATIONS) == 10
    parameters = {spec.key: spec.default for spec in VISUAL_AUGMENTATIONS}
    filters = AugmentationService.video_filters(parameters, 640, 480)
    for fragment in ("eq=", "hue=", "gblur=", "noise=", "vignette=", "crop=", "rotate="):
        assert fragment in filters


def test_visual_index_skips_orphaned_outputs(tmp_path):
    dataset = make_dataset(tmp_path / "dataset", episodes=2)
    orphan_data = dataset.root / "studio" / "episodes" / "episode_000007.parquet"
    orphan_video = dataset.root / "studio" / "videos" / "episode_000009"
    orphan_data.parent.mkdir(parents=True)
    orphan_data.touch()
    orphan_video.mkdir(parents=True)

    assert AugmentationService(dataset)._next_episode_index() == 10


def test_visual_commits_each_episode_before_next_render(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    make_dataset(root, episodes=2)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"]["observation.images.camera"] = {
        "dtype": "video",
        "shape": [3, 48, 64],
        "info": {"video.width": 64, "video.height": 48, "has_audio": False},
    }
    info_path.write_text(json.dumps(info), encoding="utf-8")
    for index in range(2):
        video = root / "videos" / "chunk-000" / "observation.images.camera" / f"episode_{index:06d}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.touch()

    dataset = LeRobotDataset(root)
    calls = 0

    def render_then_fail(segment, target, fallback_duration, parameters, width, height):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("controlled render failure")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"rendered")

    service = AugmentationService(dataset, seed=1)
    monkeypatch.setattr(service, "_render_video", render_then_fail)
    with pytest.raises(RuntimeError, match="controlled render failure"):
        service.create_visual({"brightness": 0.1}, 2)

    manifest = json.loads(dataset.augmentation_path.read_text(encoding="utf-8"))
    assert len(manifest) == 1
    assert manifest[0]["augmentation"] == "visual"
    assert (root / manifest[0]["data_file"]).exists()
    journals = list((root / "studio" / "operations").glob("visual-*.json"))
    journal = json.loads(journals[0].read_text(encoding="utf-8"))
    assert journal["status"] == "failed"
    assert journal["created"] == [2]
