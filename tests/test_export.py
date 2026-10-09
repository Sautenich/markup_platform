from __future__ import annotations

import json

from lerobot_studio.services.export import SplitExporter
from lerobot_studio.services.lerobot import LeRobotDataset
from test_augment import make_dataset


def test_split_partition_is_exhaustive(tmp_path):
    dataset = make_dataset(tmp_path / "source", episodes=10)
    parts = SplitExporter(dataset).partition(60, 20, 20, seed=3)
    indices = [episode.index for episodes in parts.values() for episode in episodes]
    assert sorted(indices) == list(range(10))
    assert [len(parts[key]) for key in ("train", "validation", "test")] == [6, 2, 2]


def test_split_export_creates_new_lerobot_datasets(tmp_path):
    dataset = make_dataset(tmp_path / "source", episodes=5)
    created = SplitExporter(dataset).export(tmp_path / "output", (60, 20, 20))
    assert len(created) == 3
    for path in created:
        assert (path / "meta" / "info.json").exists()
        assert (path / "meta" / "stats.json").exists()
        assert (path / "meta" / "episodes_stats.jsonl").exists()
        info = json.loads((path / "meta" / "info.json").read_text(encoding="utf-8"))
        exported = LeRobotDataset(path)
        assert len(exported.episodes()) == info["total_episodes"]


def test_explicit_selection_export(tmp_path):
    dataset = make_dataset(tmp_path / "source", episodes=5)
    target = tmp_path / "selection"
    SplitExporter(dataset).export_selection(target, dataset.episodes()[:2])
    exported = LeRobotDataset(target)
    assert len(exported.episodes()) == 2
    assert exported.info.total_episodes == 2
    assert len((target / "meta" / "episodes_stats.jsonl").read_text(encoding="utf-8").splitlines()) == 2
