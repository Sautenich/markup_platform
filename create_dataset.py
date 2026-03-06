import argparse
import os
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


# =========================
# Dataset generation script
# =========================

RNG = np.random.default_rng()


def choose_crop_origin(img_w: int, img_h: int, crop_size: int) -> Tuple[int, int]:
    """Return (x, y) for the top‑left corner of a square crop."""
    if img_w < crop_size or img_h < crop_size:
        raise ValueError("Image is smaller than the requested crop size.")

    max_x = img_w - crop_size
    max_y = img_h - crop_size

    x = int(RNG.integers(0, max_x + 1))
    y = int(RNG.integers(0, max_y + 1))
    return x, y


def sample_tile_center(map_size: int, tile_size: int) -> Tuple[int, int]:
    """Sample a tile center inside a square crop so that the full tile fits."""
    half = tile_size // 2
    low = half
    high = map_size - half
    cx = int(RNG.integers(low, high + 1))
    cy = int(RNG.integers(low, high + 1))
    return cx, cy


def generate_annotations(
    satellite_dir: Path,
    output_csv: Path,
    map_size: int = 1000,
    tile_size: int = 200,
    samples_per_image: int = 100,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> None:
    """
    Generate annotations.csv according to the proposed schema.

    Each row describes:
      - ID of the source satellite image
      - coordinates of a 1000x1000 crop within that image
      - current and target tile centers within the crop
      - ground‑truth vector (dx, dy) between these tiles
    """
    satellite_dir = Path(satellite_dir)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    image_paths = [
        p for p in satellite_dir.iterdir() if p.suffix.lower() in image_exts and p.is_file()
    ]

    if not image_paths:
        raise RuntimeError(f"No satellite images found in {satellite_dir}")

    records = []
    sample_id = 0

    for img_path in image_paths:
        full_map = Image.open(img_path).convert("RGB")
        w, h = full_map.size

        if w < map_size or h < map_size:
            # Пропускаем слишком маленькие карты.
            continue

        for _ in range(samples_per_image):
            try:
                crop_x, crop_y = choose_crop_origin(w, h, map_size)
            except ValueError:
                break

            # Выбираем сплит на лету
            r = RNG.random()
            if r < train_ratio:
                split = "train"
            elif r < train_ratio + val_ratio:
                split = "val"
            else:
                split = "test"

            cur_x, cur_y = sample_tile_center(map_size, tile_size)
            tar_x, tar_y = sample_tile_center(map_size, tile_size)

            dx = float(tar_x - cur_x)
            dy = float(tar_y - cur_y)

            records.append(
                {
                    "sample_id": sample_id,
                    "map_id": img_path.stem,
                    "map_crop_x": int(crop_x),
                    "map_crop_y": int(crop_y),
                    "cur_tile_x": float(cur_x),
                    "cur_tile_y": float(cur_y),
                    "tar_tile_x": float(tar_x),
                    "tar_tile_y": float(tar_y),
                    "vector_dx": dx,
                    "vector_dy": dy,
                    "split": split,
                }
            )
            sample_id += 1

    if not records:
        raise RuntimeError("No samples were generated. Check image sizes and parameters.")

    df = pd.DataFrame.from_records(records)

    df.to_csv(output_csv, index=False)

    # Также создадим отдельные CSV на всякий случай.
    root = output_csv.parent
    df[df["split"] == "train"].to_csv(root / "annotations_train.csv", index=False)
    df[df["split"] == "val"].to_csv(root / "annotations_val.csv", index=False)
    df[df["split"] == "test"].to_csv(root / "annotations_test.csv", index=False)


# =========================
# PyTorch Dataset
# =========================


class NavigationDataset(Dataset):
    def __init__(self, csv_file: str, map_dir: str, split: str | None = None, transform=None):
        """
        Args:
            csv_file: путь к CSV с аннотациями.
            map_dir: директория с исходными спутниковыми снимками.
            split: 'train' | 'val' | 'test' или None (использовать все строки).
            transform: аугментации; должны уметь обрабатывать сразу три изображения
                       (sat_img, cur_tile, tar_tile) синхронно.
        """
        self.annotations = pd.read_csv(csv_file)
        if split is not None and "split" in self.annotations.columns:
            self.annotations = self.annotations[self.annotations["split"] == split].reset_index(
                drop=True
            )

        self.map_dir = Path(map_dir)
        self.transform = transform

        self.map_size = 1000
        self.tile_size = 200

        self.to_tensor = T.ToTensor()

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int):
        row = self.annotations.iloc[idx]

        map_path = self.map_dir / f"{row['map_id']}.tif"
        if not map_path.exists():
            # Пытаемся подобрать другое расширение
            for ext in (".jpg", ".jpeg", ".png", ".tiff"):
                alt = self.map_dir / f"{row['map_id']}{ext}"
                if alt.exists():
                    map_path = alt
                    break

        full_map = Image.open(map_path).convert("RGB")

        x_start = int(row["map_crop_x"])
        y_start = int(row["map_crop_y"])

        sat_img = full_map.crop(
            (
                x_start,
                y_start,
                x_start + self.map_size,
                y_start + self.map_size,
            )
        )

        cur_cx, cur_cy = float(row["cur_tile_x"]), float(row["cur_tile_y"])
        tar_cx, tar_cy = float(row["tar_tile_x"]), float(row["tar_tile_y"])

        half = self.tile_size // 2

        cur_tile = sat_img.crop(
            (
                cur_cx - half,
                cur_cy - half,
                cur_cx + half,
                cur_cy + half,
            )
        )

        tar_tile = sat_img.crop(
            (
                tar_cx - half,
                tar_cy - half,
                tar_cx + half,
                tar_cy + half,
            )
        )

        if self.transform:
            sat_img, cur_tile, tar_tile = self.transform(sat_img, cur_tile, tar_tile)
        else:
            sat_img = self.to_tensor(sat_img)
            cur_tile = self.to_tensor(cur_tile)
            tar_tile = self.to_tensor(tar_tile)

        dx = float(row["vector_dx"])
        dy = float(row["vector_dy"])

        norm_cur_xy = torch.tensor(
            [cur_cx / self.map_size, cur_cy / self.map_size],
            dtype=torch.float32,
        )
        norm_tar_xy = torch.tensor(
            [tar_cx / self.map_size, tar_cy / self.map_size],
            dtype=torch.float32,
        )
        target_vector = torch.tensor([dx, dy], dtype=torch.float32)

        labels = {
            "cur_xy": norm_cur_xy,
            "tar_xy": norm_tar_xy,
            "vector": target_vector,
        }

        return sat_img, cur_tile, tar_tile, labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate navigation dataset annotations.")
    parser.add_argument(
        "--root",
        type=str,
        default="dataset",
        help="Корневая директория датасета (будет создана при необходимости).",
    )
    parser.add_argument(
        "--satellite-dir",
        type=str,
        default="dataset/satellite_maps",
        help="Директория с исходными спутниковыми снимками.",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="dataset/annotations.csv",
        help="Путь для сохранения основного CSV с аннотациями.",
    )
    parser.add_argument(
        "--samples-per-image",
        type=int,
        default=500,
        help="Сколько примеров генерировать на одну спутниковую карту.",
    )
    parser.add_argument(
        "--map-size",
        type=int,
        default=1000,
        help="Размер квадрата карты (пиксели).",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=200,
        help="Размер тайла (пиксели).",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(args.root)
    sat_dir = Path(args.satellite_dir)

    # Создаём базовую структуру папок (только сырые спутниковые снимки и аннотации).
    (root / "satellite_maps").mkdir(parents=True, exist_ok=True)

    generate_annotations(
        satellite_dir=sat_dir,
        output_csv=Path(args.annotations),
        map_size=args.map_size,
        tile_size=args.tile_size,
        samples_per_image=args.samples_per_image,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()

