"""Adapter that registers Studio-selected datasets without modifying the submodule.

Being-H keeps dataset paths in a Python registry.  This launcher injects the
selected LeRobot roots into that registry in every torchrun worker and then
hands control to the upstream training entry point.
"""

from __future__ import annotations

import argparse
import json
import math
import runpy
import sys
import time
from pathlib import Path


def _register_generic_config(dataset_path: Path) -> None:
    """Create a Being-H DataConfig from a LeRobot info.json schema."""
    info = json.loads((dataset_path / "meta" / "info.json").read_text(encoding="utf-8"))
    features = info.get("features", {})
    state_source = "observation.state"
    action_source = "action"
    if state_source not in features or action_source not in features:
        raise ValueError("studio_generic требует поля observation.state и action")
    state_dim = int(features[state_source]["shape"][0])
    action_dim = int(features[action_source]["shape"][0])
    if state_dim > 200 or action_dim > 200:
        raise ValueError("Being-H unified space поддерживает не более 200 state/action измерений")
    video_sources = [key for key, value in features.items() if value.get("dtype") == "video"]

    from configs import data_config
    from configs.data_config import BaseDataConfig, ModalityDef
    from BeingH.dataset.transform.base import ComposedModalityTransform
    from BeingH.dataset.transform.state_action import StateActionToTensor, StateActionTransform

    class StudioGenericDataConfig(BaseDataConfig):
        VIDEO_KEYS = [f"video.camera_{index}" for index in range(len(video_sources))]
        VIDEO_SOURCE_COLUMNS = dict(zip(VIDEO_KEYS, video_sources, strict=True))
        STATE_KEYS = ["state.joints"]
        ACTION_KEYS = ["action.joints"]
        LANGUAGE_KEYS = ["language.instruction"]
        UNIFIED_MAPPING = {
            "state.joints": (0, state_dim),
            "action.joints": (0, action_dim),
        }
        state_normalization_modes = {"state.joints": "min_max"}
        action_normalization_modes = {"action.joints": "min_max"}

        def define_modalities(self):
            modalities = {
                "language.instruction": ModalityDef(source_column="task_index", start=0, end=0),
                "state.joints": ModalityDef(source_column=state_source, start=0, end=state_dim),
                "action.joints": ModalityDef(
                    source_column=action_source, start=0, end=action_dim, absolute=True
                ),
            }
            return self.add_video_modality(modalities)

        def get_feature_meta(self):
            return {
                "state.joints": (f"{state_dim}-d robot joint state", state_dim),
                "action.joints": (f"{action_dim}-d robot joint action", action_dim),
            }

        def get_transforms(self):
            return ComposedModalityTransform(
                [
                    StateActionToTensor(apply_to=self.STATE_KEYS),
                    StateActionTransform(
                        apply_to=self.STATE_KEYS,
                        normalization_modes=self.state_normalization_modes,
                    ),
                    StateActionToTensor(apply_to=self.ACTION_KEYS),
                    StateActionTransform(
                        apply_to=self.ACTION_KEYS,
                        normalization_modes=self.action_normalization_modes,
                    ),
                ]
            )

    data_config.DATA_CONFIG_MAP["studio_generic"] = StudioGenericDataConfig


def _write_dataset_yaml(spec_path: Path, datasets: list[str], config_name: str) -> Path:
    # PyYAML is an upstream Being-H dependency, but writing this small document
    # ourselves also keeps configuration generation inspectable.
    names = [f"studio_dataset_{index}" for index in range(len(datasets))]
    lines = ["studio_posttrain:", "  dataset_names:"]
    lines.extend(f"  - {name}" for name in names)
    lines.append("  data_config_names:")
    lines.extend(f"  - {config_name}" for _ in names)
    lines.append("  embodiment_tags:")
    lines.extend("  - new_embodiment" for _ in names)
    lines.extend(
        [
            '  sampling_strategy: "step"',
            '  video_backend: "torchvision_av"',
            "  vit_transform_args:",
            '    type: "beingh"',
            '    normalize_type: "imagenet"',
            "    use_color_jitter: true",
            "  frame_step_size:",
        ]
    )
    lines.extend("  - 1" for _ in names)
    lines.append("  num_used_episodes_per_task:")
    lines.extend("  - -1" for _ in names)
    lines.append("  num_used_episodes_per_dataset:")
    lines.extend("  - -1" for _ in names)
    lines.append("  num_used_frames_per_dataset:")
    lines.extend("  - -1" for _ in names)
    lines.append("  weight: 1")
    path = spec_path.with_name("studio_datasets.yaml")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _train(spec_path: Path) -> int:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    params = spec["hyperparameters"]
    datasets = spec["datasets"]

    from configs import dataset_info
    from BeingH.dataset.datasets.vla_dataset import LeRobotIterableDataset

    if str(params["data_config_name"]) == "studio_generic":
        schemas = []
        for dataset_path in datasets:
            info = json.loads((Path(dataset_path) / "meta" / "info.json").read_text(encoding="utf-8"))
            features = info.get("features", {})
            schemas.append(
                (
                    features.get("observation.state", {}).get("shape"),
                    features.get("action", {}).get("shape"),
                    [key for key, value in features.items() if value.get("dtype") == "video"],
                )
            )
        if any(schema != schemas[0] for schema in schemas[1:]):
            raise ValueError(
                "Для совместного Being-H-обучения датасеты должны иметь одинаковую state/action/video схему"
            )
        _register_generic_config(Path(datasets[0]))

    dataset_info.DATASET_REGISTRY["studio_posttrain"] = LeRobotIterableDataset
    dataset_info.DATASET_INFO["studio_posttrain"] = {
        f"studio_dataset_{index}": {"dataset_path": path}
        for index, path in enumerate(datasets)
    }
    yaml_path = _write_dataset_yaml(spec_path, datasets, str(params["data_config_name"]))
    output = Path(spec["output_dir"])
    sys.argv = [
        "BeingH/train/train.py",
        "--mllm_path",
        str(params["mllm_path"]),
        "--expert_path",
        str(params["expert_path"]),
        "--resume_from",
        str(params["resume_from"]),
        "--resume_model_only",
        "True",
        "--layer_module",
        "Qwen3MoTDecoderLayer",
        "--use_expert",
        "True",
        "--use_flow_matching",
        "True",
        "--llm_qk_norm",
        "True",
        "--action_chunk_length",
        str(params["action_chunk_length"]),
        "--dataset_config_file",
        str(yaml_path),
        "--save_merged_metadata",
        "True",
        "--conv_style",
        "being_h0",
        "--prompt_template",
        "long",
        "--output_dir",
        str(output),
        "--logging_dir",
        str(output / "logs"),
        "--num_workers",
        str(params["num_workers"]),
        "--max_steps",
        str(params["max_steps"]),
        "--save_steps",
        str(params["save_steps"]),
        "--logging_steps",
        "10",
        "--learning_rate",
        str(params["learning_rate"]),
        "--weight_decay",
        str(params["weight_decay"]),
        "--warmup_ratio",
        str(params["warmup_ratio"]),
        "--lr_scheduler",
        "cosine",
        "--gradient_accumulation_steps",
        "1",
    ]
    runpy.run_module("BeingH.train.train", run_name="__main__")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    import cv2
    import numpy as np

    from lerobot_studio.services.lerobot import LeRobotDataset

    dataset = LeRobotDataset(args.dataset)
    episodes = [item for item in dataset.episodes() if not item.augmented]
    if not episodes:
        raise ValueError("В датасете нет эпизодов для тестирования")
    episode = episodes[0]
    frame = dataset.dataframe(episode)
    manifest: dict = {}
    for parent in [args.checkpoint, *list(args.checkpoint.parents)[:3]]:
        candidate = parent / "studio_checkpoint.json"
        if candidate.exists():
            manifest = json.loads(candidate.read_text(encoding="utf-8"))
            break
    params = manifest.get("hyperparameters", {})
    config_name = str(params.get("data_config_name", "studio_generic"))
    if config_name == "studio_generic":
        _register_generic_config(args.dataset)

    from BeingH.inference.beingh_policy import BeingHPolicy
    from BeingH.utils.constants import INSTRUCTION_TEMPLATE

    policy = BeingHPolicy(
        model_path=str(args.checkpoint),
        data_config_name=config_name,
        dataset_name="studio_posttrain",
        embodiment_tag="new_embodiment",
        instruction_template=INSTRUCTION_TEMPLATE,
        enable_rtc=False,
    )
    captures: dict[str, cv2.VideoCapture] = {}
    try:
        for segment in episode.videos:
            capture = cv2.VideoCapture(str(segment.path))
            if not capture.isOpened():
                raise RuntimeError(f"Не удалось открыть видео: {segment.path}")
            captures[segment.key] = capture
        total = min(16, len(frame))
        indices = np.linspace(0, len(frame) - 1, total, dtype=int)
        errors: list[np.ndarray] = []
        started = time.monotonic()
        for position, frame_index in enumerate(indices, 1):
            row = frame.iloc[int(frame_index)]
            observations = {
                "state.joints": np.asarray(row["observation.state"], dtype=np.float32),
                "language.instruction": episode.prompt or "Execute the demonstrated task",
            }
            for video_index, segment in enumerate(episode.videos):
                capture = captures[segment.key]
                timestamp = segment.start_s + float(frame_index) / dataset.info.fps
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError(f"Не удалось прочитать кадр {frame_index}: {segment.path}")
                observations[f"video.camera_{video_index}"] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            prediction = policy.get_action(observations)
            predicted_action = np.asarray(prediction["action.joints"], dtype=np.float32)[0]
            target_action = np.asarray(row["action"], dtype=np.float32)
            errors.append(predicted_action - target_action)
            joined = np.concatenate(errors)
            mae = float(np.mean(np.abs(joined)))
            rmse = float(math.sqrt(np.mean(np.square(joined))))
            elapsed = time.monotonic() - started
            eta = elapsed / position * (total - position)
            print(
                f"step {position}/{total} [METRIC] mae={mae:.6f} rmse={rmse:.6f} eta_seconds={eta:.1f}",
                flush=True,
            )
    finally:
        for capture in captures.values():
            capture.release()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"action_mae": mae, "action_rmse": rmse, "samples": total}
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Action MAE: {mae:.6f}")
    print(f"Action RMSE: {rmse:.6f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", nargs="?", type=Path)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.evaluate:
        return _evaluate(args)
    if args.spec is None:
        parser.error("spec is required for training")
    return _train(args.spec)


if __name__ == "__main__":
    raise SystemExit(main())
