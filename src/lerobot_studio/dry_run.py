from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path


def _train(args: argparse.Namespace) -> int:
    output = args.output_dir.resolve()
    checkpoint = output / "base_checkpoint"
    checkpoint.mkdir(parents=True, exist_ok=True)
    total = max(1, min(args.max_steps, 20))
    print(f"[dry-run] Модель {args.model}: проверка конфигурации", flush=True)
    print("[dry-run] CUDA не используется, веса не изменяются", flush=True)
    for step in range(1, total + 1):
        progress = step / total
        loss = 1.4 * math.exp(-2.5 * progress) + random.uniform(0.01, 0.05)
        eta = (total - step) * 0.15
        print(
            f"step {step}/{total} loss={loss:.5f} "
            f"[METRIC] loss={loss:.5f} eta_seconds={eta:.1f}",
            flush=True,
        )
        time.sleep(0.15)
    metadata = {
        "model": args.model,
        "kind": "base_checkpoint",
        "base_checkpoint": args.base_checkpoint,
        "dry_run": True,
    }
    (checkpoint / "studio_checkpoint.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[dry-run] Base checkpoint сохранён: {checkpoint}", flush=True)
    return 0


def _test(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[dry-run] Тестирование {args.model}; инференс модели отключён", flush=True)
    mae = 0.0
    rmse = 0.0
    for step in range(1, 11):
        mae = random.uniform(0.025, 0.18)
        rmse = mae * random.uniform(1.08, 1.65)
        eta = (10 - step) * 0.15
        print(
            f"trajectory {step}/10 [METRIC] mae={mae:.5f} rmse={rmse:.5f} eta_seconds={eta:.1f}",
            flush=True,
        )
        time.sleep(0.15)
    metrics = {"action_mae": mae, "action_rmse": rmse, "dry_run": True}
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Action MAE: {mae:.5f}", flush=True)
    print(f"Action RMSE: {rmse:.5f}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Безопасная симуляция заданий LeRobot Studio")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--model", required=True)
    train.add_argument("--output-dir", required=True, type=Path)
    train.add_argument("--max-steps", type=int, default=20)
    train.add_argument("--base-checkpoint", default="base")
    test = subparsers.add_parser("test")
    test.add_argument("--model", required=True)
    test.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    return _train(args) if args.mode == "train" else _test(args)


if __name__ == "__main__":
    raise SystemExit(main())
