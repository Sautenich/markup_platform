from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from lerobot_studio.config import config_dir


GR00T = "gr00t"
BEING_H05 = "being_h05"

MODEL_NAMES = {
    GR00T: "NVIDIA GR00T N1.7",
    BEING_H05: "Being-H0.5",
}


DEFAULT_PROFILES: dict[str, dict[str, dict[str, Any]]] = {
    GR00T: {
        "Быстрый": {
            "base_model_path": "nvidia/GR00T-N1.7-3B",
            "embodiment_tag": "NEW_EMBODIMENT",
            "modality_config_path": "studio_generic",
            "max_steps": 500,
            "global_batch_size": 8,
            "learning_rate": 0.0001,
            "weight_decay": 0.00001,
            "warmup_ratio": 0.05,
            "save_steps": 500,
            "num_gpus": 1,
            "dataloader_num_workers": 2,
        },
        "Сбалансированный": {
            "base_model_path": "nvidia/GR00T-N1.7-3B",
            "embodiment_tag": "NEW_EMBODIMENT",
            "modality_config_path": "studio_generic",
            "max_steps": 2000,
            "global_batch_size": 32,
            "learning_rate": 0.0001,
            "weight_decay": 0.00001,
            "warmup_ratio": 0.05,
            "save_steps": 500,
            "num_gpus": 1,
            "dataloader_num_workers": 4,
        },
        "Точное обучение": {
            "base_model_path": "nvidia/GR00T-N1.7-3B",
            "embodiment_tag": "NEW_EMBODIMENT",
            "modality_config_path": "studio_generic",
            "max_steps": 10000,
            "global_batch_size": 64,
            "learning_rate": 0.00005,
            "weight_decay": 0.00001,
            "warmup_ratio": 0.05,
            "save_steps": 1000,
            "num_gpus": 1,
            "dataloader_num_workers": 8,
        },
    },
    BEING_H05: {
        "Быстрый": {
            "python_executable": "python",
            "mllm_path": "InternVL3_5-2B",
            "expert_path": "Qwen3-0.6B",
            "resume_from": "Being-H05-2B",
            "data_config_name": "studio_generic",
            "max_steps": 500,
            "learning_rate": 0.0001,
            "weight_decay": 0.00001,
            "warmup_ratio": 0.05,
            "save_steps": 500,
            "action_chunk_length": 16,
            "num_gpus": 1,
            "num_workers": 4,
        },
        "Сбалансированный": {
            "python_executable": "python",
            "mllm_path": "InternVL3_5-2B",
            "expert_path": "Qwen3-0.6B",
            "resume_from": "Being-H05-2B",
            "data_config_name": "studio_generic",
            "max_steps": 10000,
            "learning_rate": 0.0001,
            "weight_decay": 0.00001,
            "warmup_ratio": 0.05,
            "save_steps": 2000,
            "action_chunk_length": 16,
            "num_gpus": 1,
            "num_workers": 8,
        },
        "Точное обучение": {
            "python_executable": "python",
            "mllm_path": "InternVL3_5-2B",
            "expert_path": "Qwen3-0.6B",
            "resume_from": "Being-H05-2B",
            "data_config_name": "studio_generic",
            "max_steps": 60000,
            "learning_rate": 0.00005,
            "weight_decay": 0.00001,
            "warmup_ratio": 0.05,
            "save_steps": 10000,
            "action_chunk_length": 16,
            "num_gpus": 4,
            "num_workers": 12,
        },
    },
}


@dataclass(slots=True)
class JobSpec:
    mode: str
    model: str
    datasets: list[str]
    output_dir: str
    hyperparameters: dict[str, Any]
    program: str
    arguments: list[str]
    working_directory: str
    dry_run: bool = False
    checkpoint: str | None = None

    def public_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("program", None)
        payload.pop("arguments", None)
        return payload


class ProfileStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or config_dir() / "model_profiles.json"

    def _custom(self) -> dict[str, dict[str, dict[str, Any]]]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def profiles(self, model: str) -> dict[str, dict[str, Any]]:
        result = {name: dict(values) for name, values in DEFAULT_PROFILES[model].items()}
        result.update({name: dict(values) for name, values in self._custom().get(model, {}).items()})
        return result

    def save(self, model: str, name: str, values: dict[str, Any]) -> None:
        custom = self._custom()
        custom.setdefault(model, {})[name] = values
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(custom, ensure_ascii=False, indent=2), encoding="utf-8")


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def default_output_dir(workspace: Path, model: str) -> Path:
    return workspace / "checkpoints" / model / _timestamp()


def _uv_or_python() -> tuple[str, list[str]]:
    uv = shutil.which("uv")
    return (uv, ["run", "python"]) if uv else (sys.executable, [])


def _prepare_gr00t_datasets(datasets: list[Path], output_dir: Path) -> tuple[list[str], Path]:
    """Create non-destructive dataset views with a generic GR00T modality schema."""
    runtime_root = output_dir / "studio_gr00t_runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    schemas: list[tuple[int, int, list[str]]] = []
    prepared: list[str] = []
    for index, source in enumerate(datasets):
        source = source.resolve()
        info = json.loads((source / "meta" / "info.json").read_text(encoding="utf-8"))
        features = info.get("features", {})
        if "observation.state" not in features or "action" not in features:
            raise ValueError("GR00T studio_generic требует поля observation.state и action")
        state_dim = int(features["observation.state"]["shape"][0])
        action_dim = int(features["action"]["shape"][0])
        video_keys = [key for key, value in features.items() if value.get("dtype") == "video"]
        schemas.append((state_dim, action_dim, video_keys))
        target = runtime_root / f"dataset_{index}_{source.name}"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source / "meta", target / "meta", dirs_exist_ok=True)
        for directory in ("data", "videos"):
            source_item = source / directory
            target_item = target / directory
            if source_item.exists() and not target_item.exists():
                target_item.symlink_to(source_item, target_is_directory=True)
        modality = {
            "state": {"joints": {"start": 0, "end": state_dim}},
            "action": {"joints": {"start": 0, "end": action_dim}},
            "video": {
                f"camera_{video_index}": {"original_key": key}
                for video_index, key in enumerate(video_keys)
            },
            "annotation": {
                "human.task_description": {"original_key": "task_index"}
            },
        }
        (target / "meta" / "modality.json").write_text(
            json.dumps(modality, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        prepared.append(str(target))
    reference = schemas[0]
    if any(schema != reference for schema in schemas[1:]):
        raise ValueError("Для совместного GR00T-обучения датасеты должны иметь одинаковую state/action/video схему")
    state_dim, action_dim, video_keys = reference
    config_path = runtime_root / "studio_modality_config.py"
    video_modalities = [f"camera_{index}" for index in range(len(video_keys))]
    config_path.write_text(
        "\n".join(
            [
                "from gr00t.configs.data.embodiment_configs import register_modality_config",
                "from gr00t.data.embodiment_tags import EmbodimentTag",
                "from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig",
                "studio_config = {",
                f"    'video': ModalityConfig(delta_indices=[0], modality_keys={video_modalities!r}),",
                "    'state': ModalityConfig(delta_indices=[0], modality_keys=['joints']),",
                "    'action': ModalityConfig(delta_indices=list(range(16)), modality_keys=['joints'], action_configs=[ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)]),",
                "    'language': ModalityConfig(delta_indices=[0], modality_keys=['annotation.human.task_description']),",
                "}",
                "register_modality_config(studio_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return prepared, config_path


def build_train_job(
    workspace: Path,
    model: str,
    datasets: list[Path],
    output_dir: Path,
    hyperparameters: dict[str, Any],
    dry_run: bool,
) -> JobSpec:
    output_dir = output_dir.expanduser().resolve()
    paths = [str(path.resolve()) for path in datasets]
    if dry_run:
        program = sys.executable
        arguments = [
            "-m",
            "lerobot_studio.dry_run",
            "train",
            "--model",
            model,
            "--output-dir",
            str(output_dir),
            "--max-steps",
            str(hyperparameters.get("max_steps", 100)),
            "--base-checkpoint",
            str(hyperparameters.get("base_model_path") or hyperparameters.get("resume_from") or "base"),
        ]
        cwd = workspace
    elif model == GR00T:
        root = workspace / "third_party" / "Isaac-GR00T"
        command_paths = paths
        modality = str(hyperparameters.get("modality_config_path", "")).strip()
        if modality == "studio_generic":
            command_paths, generated_config = _prepare_gr00t_datasets(datasets, output_dir)
            modality = str(generated_config)
        program, prefix = _uv_or_python()
        arguments = prefix + [
            "gr00t/experiment/launch_finetune.py",
            "--base-model-path",
            str(hyperparameters["base_model_path"]),
            "--dataset-path",
            os.pathsep.join(command_paths),
            "--embodiment-tag",
            str(hyperparameters["embodiment_tag"]),
            "--num-gpus",
            str(hyperparameters["num_gpus"]),
            "--output-dir",
            str(output_dir),
            "--max-steps",
            str(hyperparameters["max_steps"]),
            "--global-batch-size",
            str(hyperparameters["global_batch_size"]),
            "--learning-rate",
            str(hyperparameters["learning_rate"]),
            "--weight-decay",
            str(hyperparameters["weight_decay"]),
            "--warmup-ratio",
            str(hyperparameters["warmup_ratio"]),
            "--save-steps",
            str(hyperparameters["save_steps"]),
            "--dataloader-num-workers",
            str(hyperparameters["dataloader_num_workers"]),
        ]
        if modality:
            arguments += ["--modality-config-path", modality]
        cwd = root
    else:
        root = workspace / "third_party" / "Being-H" / "Being-H05"
        output_dir.mkdir(parents=True, exist_ok=True)
        runner_spec = output_dir / "being_h_job.json"
        runner_spec.write_text(
            json.dumps(
                {"datasets": paths, "output_dir": str(output_dir), "hyperparameters": hyperparameters},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        configured_python = str(hyperparameters.get("python_executable", "python"))
        python = shutil.which(configured_python) or configured_python
        gpu_count = int(hyperparameters.get("num_gpus", 1))
        if gpu_count > 1:
            program = python
            arguments = [
                "-m",
                "torch.distributed.run",
                "--nproc_per_node",
                str(gpu_count),
                "-m",
                "lerobot_studio.services.being_h_runner",
                str(runner_spec),
            ]
        else:
            program = python
            arguments = ["-m", "lerobot_studio.services.being_h_runner", str(runner_spec)]
        cwd = root
    return JobSpec("train", model, paths, str(output_dir), hyperparameters, program, arguments, str(cwd), dry_run)


def build_test_job(
    workspace: Path,
    model: str,
    dataset: Path,
    checkpoint: Path,
    output_dir: Path,
    dry_run: bool,
) -> JobSpec:
    output_dir = output_dir.expanduser().resolve()
    if dry_run:
        program = sys.executable
        arguments = [
            "-m",
            "lerobot_studio.dry_run",
            "test",
            "--model",
            model,
            "--output-dir",
            str(output_dir),
        ]
        cwd = workspace
    elif model == GR00T:
        root = workspace / "third_party" / "Isaac-GR00T"
        dataset_path = dataset.resolve()
        if not (dataset_path / "meta" / "modality.json").exists():
            prepared, _generated_config = _prepare_gr00t_datasets([dataset_path], output_dir)
            dataset_path = Path(prepared[0])
        program, prefix = _uv_or_python()
        arguments = prefix + [
            "gr00t/eval/open_loop_eval.py",
            "--dataset-path",
            str(dataset_path),
            "--embodiment-tag",
            "NEW_EMBODIMENT",
            "--model-path",
            str(checkpoint.expanduser().resolve()),
            "--traj-ids",
            "0",
            "--execution-horizon",
            "16",
            "--save-plot-path",
            str(output_dir),
        ]
        cwd = root
    else:
        root = workspace / "third_party" / "Being-H" / "Being-H05"
        metadata = checkpoint_metadata(checkpoint)
        configured_python = str(metadata.get("hyperparameters", {}).get("python_executable", "python"))
        program = shutil.which(configured_python) or configured_python
        arguments = [
            "-m",
            "lerobot_studio.services.being_h_runner",
            "--evaluate",
            "--checkpoint",
            str(checkpoint.expanduser().resolve()),
            "--dataset",
            str(dataset.resolve()),
            "--output-dir",
            str(output_dir),
        ]
        cwd = root
    return JobSpec(
        "test",
        model,
        [str(dataset.resolve())],
        str(output_dir),
        {},
        program,
        arguments,
        str(cwd),
        dry_run,
        str(checkpoint.expanduser().resolve()),
    )


def discover_checkpoints(workspace: Path) -> list[Path]:
    root = workspace / "checkpoints"
    if not root.exists():
        return []
    found: set[Path] = set()
    for manifest in root.rglob("studio_checkpoint.json"):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if payload.get("dry_run") and (manifest.parent / "base_checkpoint" / "studio_checkpoint.json").exists():
            continue
        found.add(manifest.parent)
    for candidate in root.rglob("checkpoint-*"):
        if candidate.is_dir():
            found.add(candidate)
    return sorted(found, key=lambda item: item.stat().st_mtime, reverse=True)


def checkpoint_model(path: Path) -> str | None:
    value = checkpoint_metadata(path).get("model")
    return value if value in MODEL_NAMES else None


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    candidates = [parent / "studio_checkpoint.json" for parent in [path, *list(path.parents)[:3]]]
    for manifest in candidates:
        try:
            return json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return {}
