from __future__ import annotations

import time

from lerobot_studio.ui.dialogs import GraphDialog
from lerobot_studio.ui.episode_page import EpisodePage
from test_augment import make_dataset


def test_graph_dialog_loads_data_without_blocking_gui(qtbot, tmp_path, monkeypatch):
    dataset = make_dataset(tmp_path / "dataset", episodes=1)
    original_dataframe = dataset.dataframe

    def slow_dataframe(episode):
        time.sleep(0.25)
        return original_dataframe(episode)

    monkeypatch.setattr(dataset, "dataframe", slow_dataframe)
    started = time.monotonic()
    dialog = GraphDialog(dataset, dataset.episodes()[0])
    construction_time = time.monotonic() - started
    qtbot.addWidget(dialog)
    dialog.show()

    assert construction_time < 0.15
    assert dialog.isVisible()
    qtbot.waitUntil(lambda: dialog.loaded, timeout=3000)


def test_episode_graph_window_is_non_modal(qtbot, tmp_path):
    page = EpisodePage()
    qtbot.addWidget(page)
    page.open_dataset(make_dataset(tmp_path / "dataset", episodes=1))
    page.show()

    page.show_graphs()

    assert len(page.graph_dialogs) == 1
    assert not page.graph_dialogs[0].isModal()
    assert page.graph_dialogs[0].isVisible()
    qtbot.waitUntil(lambda: page.graph_dialogs[0].loaded, timeout=3000)
