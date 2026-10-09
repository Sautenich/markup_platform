from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from lerobot_studio.services.model_jobs import (
    DEFAULT_PROFILES,
    GR00T,
    ProfileStore,
    build_train_job,
    discover_checkpoints,
)


def test_custom_hyperparameter_profile_is_persisted(tmp_path):
    store = ProfileStore(tmp_path / "profiles.json")
    values = dict(DEFAULT_PROFILES[GR00T]["Быстрый"])
    values["max_steps"] = 42
    store.save(GR00T, "Мой профиль", values)
    assert store.profiles(GR00T)["Мой профиль"]["max_steps"] == 42


def test_gr00t_job_passes_multiple_datasets(tmp_path):
    datasets = [tmp_path / "one", tmp_path / "two"]
    parameters = dict(DEFAULT_PROFILES[GR00T]["Быстрый"])
    parameters["modality_config_path"] = "custom_config.py"
    job = build_train_job(
        tmp_path,
        GR00T,
        datasets,
        tmp_path / "output",
        parameters,
        dry_run=False,
    )
    dataset_argument = job.arguments[job.arguments.index("--dataset-path") + 1]
    assert dataset_argument == os.pathsep.join(str(path.resolve()) for path in datasets)
    assert "gr00t/experiment/launch_finetune.py" in job.arguments


def test_dry_run_saves_only_base_checkpoint(tmp_path):
    output = tmp_path / "checkpoints" / "gr00t" / "run"
    job = build_train_job(
        tmp_path,
        GR00T,
        [tmp_path / "dataset"],
        output,
        DEFAULT_PROFILES[GR00T]["Быстрый"],
        dry_run=True,
    )
    result = subprocess.run([job.program, *job.arguments], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    manifest = output / "base_checkpoint" / "studio_checkpoint.json"
    assert json.loads(manifest.read_text(encoding="utf-8"))["kind"] == "base_checkpoint"
    assert discover_checkpoints(tmp_path) == [output / "base_checkpoint"]
