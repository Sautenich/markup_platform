import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Create a dataset of 2000x2000 map tiles cropped from a single big .tif "
            "(e.g. big_sample5_17.tif), and write a CUR-localization annotations CSV.\n"
            "\n"
            "Each saved image is exactly 2000x2000, so annotations use map_crop_x=map_crop_y=0."
        )
    )
    p.add_argument("--big-image", type=str, default="big_sample5_17.tif", help="Path to big .tif.")
    p.add_argument(
        "--out-dir",
        type=str,
        default="dataset/satellite_maps/2000x2000test_big_sample",
        help="Directory to save 2000x2000 tiles.",
    )
    p.add_argument(
        "--output-csv",
        type=str,
        default="dataset/annotations_big_cur_2000.csv",
        help="Output annotations CSV path.",
    )
    p.add_argument("--count", type=int, default=100, help="How many 2000x2000 tiles to generate.")
    p.add_argument("--map-size", type=int, default=2000, help="Tile map size (pixels).")
    p.add_argument("--tile-size", type=int, default=200, help="CUR tile window size (pixels).")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument(
        "--image-format",
        type=str,
        default="png",
        choices=["png", "jpg", "tif", "tiff"],
        help="Format for saved 2000x2000 tiles.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))

    big_path = Path(args.big_image)
    out_dir = Path(args.out_dir)
    out_csv = Path(args.output_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not big_path.exists():
        raise FileNotFoundError(f"Big image not found: {big_path}")

    map_size = int(args.map_size)
    tile_size = int(args.tile_size)
    half = tile_size // 2

    Image.MAX_IMAGE_PIXELS = None
    big_img = Image.open(big_path).convert("RGB")
    w, h = big_img.size
    if w < map_size or h < map_size:
        raise ValueError(f"Big image {w}x{h} is smaller than map_size {map_size}.")

    records: list[dict] = []

    for i in range(int(args.count)):
        # random crop origin
        x0 = int(rng.integers(0, (w - map_size) + 1))
        y0 = int(rng.integers(0, (h - map_size) + 1))

        tile_img = big_img.crop((x0, y0, x0 + map_size, y0 + map_size))

        map_id = f"{big_path.stem}_2000_{i:04d}"
        out_path = out_dir / f"{map_id}.{args.image_format}"
        tile_img.save(out_path)

        # choose split
        r = float(rng.random())
        if r < args.train_ratio:
            split = "train"
        elif r < args.train_ratio + args.val_ratio:
            split = "val"
        else:
            split = "test"

        # random CUR center inside 2000x2000 so that 200x200 window fits
        cur_x = int(rng.integers(half, (map_size - half) + 1))
        cur_y = int(rng.integers(half, (map_size - half) + 1))

        records.append(
            {
                "sample_id": i,
                "map_id": map_id,
                "map_crop_x": 0,
                "map_crop_y": 0,
                "cur_tile_x": float(cur_x),
                "cur_tile_y": float(cur_y),
                "tar_tile_x": float(cur_x),  # not used, keep compatible shape
                "tar_tile_y": float(cur_y),
                "vector_dx": 0.0,
                "vector_dy": 0.0,
                "split": split,
                "source_image": str(big_path.name),
                "origin_x": int(x0),
                "origin_y": int(y0),
                "map_size": map_size,
                "tile_size": tile_size,
            }
        )

    df = pd.DataFrame.from_records(records)
    df.to_csv(out_csv, index=False)

    print(f"Saved tiles: {out_dir}  (count={len(df)})")
    print(f"Saved annotations: {out_csv}")


if __name__ == "__main__":
    main()

