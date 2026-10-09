from __future__ import annotations

from test_augment import make_dataset


def test_reads_v2_episodes(tmp_path):
    dataset = make_dataset(tmp_path / "dataset", episodes=2)
    episodes = dataset.episodes()
    assert [item.index for item in episodes] == [0, 1]
    assert all(item.length == 5 for item in episodes)


def test_edits_are_persistent(tmp_path):
    dataset = make_dataset(tmp_path / "dataset", episodes=1)
    dataset.edit_episode(0, "new prompt", {"quality": "good"})
    episode = dataset.episodes(refresh=True)[0]
    assert episode.prompt == "new prompt"
    assert episode.metadata["quality"] == "good"


def test_soft_delete_exports_and_hides_episode(tmp_path):
    dataset = make_dataset(tmp_path / "dataset", episodes=2)
    destination = dataset.soft_delete(dataset.episodes()[0], tmp_path / "trash")
    assert (destination / "episode.parquet").exists()
    assert (destination / "metadata.json").exists()
    assert [episode.index for episode in dataset.episodes(refresh=True)] == [1]
