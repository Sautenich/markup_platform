import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


RNG = np.random.default_rng()


def choose_crop_origin(img_w: int, img_h: int, crop_size: int) -> tuple[int, int]:
    if img_w < crop_size or img_h < crop_size:
        raise ValueError("Image is smaller than the requested crop size.")
    x = int(RNG.integers(0, (img_w - crop_size) + 1))
    y = int(RNG.integers(0, (img_h - crop_size) + 1))
    return x, y


def sample_tile_center(map_size: int, tile_size: int) -> tuple[int, int]:
    half = tile_size // 2
    cx = int(RNG.integers(half, (map_size - half) + 1))
    cy = int(RNG.integers(half, (map_size - half) + 1))
    return cx, cy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Create NavigationDataset-compatible annotations from a single big .tif image "
            "(e.g. big_sample5_17.tif) for CUR-localization training + interactive navigation.\n"
            "\n"
            "Important: NavigationDataset currently uses fixed map_size=1000 and tile_size=200. "
            "This script defaults to the same values.\n"
            "\n"
            "Instead of sampling crops from one huge map at runtime, this script can SPLIT the big image "
            "into many 1000x1000 map tiles saved into map-dir. Annotations then use map_crop_x=map_crop_y=0.\n"
            "\n"
            "To handle different resolutions/scales, it can build such tiles for multiple scales "
            "(different map_id values)."
        )
    )
    p.add_argument("--big-image", type=str, default="big_sample5_17.tif", help="Path to big .tif.")
    p.add_argument(
        "--map-dir",
        type=str,
        default="dataset/satellite_maps",
        help="Where to write scaled images (NavigationDataset reads maps from here).",
    )
    p.add_argument(
        "--output",
        type=str,
        default="dataset/annotations_big_cur.csv",
        help="Where to save the generated annotations CSV.",
    )
    p.add_argument("--map-size", type=int, default=1000, help="Crop size in pixels (must match NavigationDataset).")
    p.add_argument("--tile-size", type=int, default=200, help="Tile size in pixels (must match NavigationDataset).")
    p.add_argument(
        "--scales",
        type=str,
        default="1.0,0.5,0.25",
        help="Comma-separated downsample scales to generate, e.g. '1.0,0.5,0.25'.",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1000,
        help="Stride for tiling the (scaled) big image into map tiles. 1000 => no overlap.",
    )
    p.add_argument(
        "--max-tiles",
        type=int,
        default=0,
        help="Optional limit on number of saved map tiles per scale (0 => no limit).",
    )
    p.add_argument(
        "--samples-per-tile",
        type=int,
        default=1,
        help="How many annotation rows to generate per saved 1000x1000 map tile.",
    )
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    p.add_argument(
        "--image-format",
        type=str,
        default="png",
        choices=["png", "jpg", "tif", "tiff"],
        help="Format for saved scaled maps inside map-dir.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    global RNG
    RNG = np.random.default_rng(int(args.seed))

    big_path = Path(args.big_image)
    map_dir = Path(args.map_dir)
    out_csv = Path(args.output)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)

    if not big_path.exists():
        raise FileNotFoundError(f"Big image not found: {big_path}")

    map_size = int(args.map_size)
    tile_size = int(args.tile_size)
    if map_size != 1000 or tile_size != 200:
        print(
            "WARNING: NavigationDataset currently uses fixed map_size=1000 and tile_size=200.\n"
            "If you change --map-size/--tile-size here, training/navigation scripts using "
            "NavigationDataset will NOT automatically pick it up."
        )

    scales = []
    for s in str(args.scales).split(","):
        s = s.strip()
        if not s:
            continue
        scales.append(float(s))
    if not scales:
        raise ValueError("No scales provided.")

    Image.MAX_IMAGE_PIXELS = None
    big_img = Image.open(big_path).convert("RGB")
    w0, h0 = big_img.size

    records: list[dict] = []
    sample_id = 0

    for scale in scales:
        if scale <= 0:
            raise ValueError(f"Invalid scale: {scale}")

        scale_tag = f"s{int(round(scale * 1000))}"
        new_w = int(round(w0 * scale))
        new_h = int(round(h0 * scale))
        if new_w < map_size or new_h < map_size:
            print(f"Skipping scale {scale}: scaled image {new_w}x{new_h} < map_size {map_size}")
            continue

        scaled_img = big_img if abs(scale - 1.0) < 1e-9 else big_img.resize((new_w, new_h), resample=Image.BICUBIC)
        w, h = scaled_img.size

        stride = int(args.stride)
        if stride <= 0:
            raise ValueError("--stride must be > 0")

        # Generate tile top-left positions on a grid.
        xs = list(range(0, w - map_size + 1, stride))
        ys = list(range(0, h - map_size + 1, stride))
        if not xs or not ys:
            print(f"Skipping scale {scale}: cannot place even one {map_size}x{map_size} tile.")
            continue

        tile_positions = [(x, y) for y in ys for x in xs]
        if args.max_tiles and int(args.max_tiles) > 0:
            tile_positions = tile_positions[: int(args.max_tiles)]

        for tile_idx, (x0, y0) in enumerate(tile_positions):
            map_id = f"{big_path.stem}_{scale_tag}_x{x0}_y{y0}"
            out_tile_path = map_dir / f"{map_id}.{args.image_format}"

            if not out_tile_path.exists():
                tile_img = scaled_img.crop((x0, y0, x0 + map_size, y0 + map_size))
                tile_img.save(out_tile_path)

            for _ in range(int(args.samples_per_tile)):
                crop_x, crop_y = 0, 0  # the saved map is already a 1000x1000 crop

                r = RNG.random()
                if r < args.train_ratio:
                    split = "train"
                elif r < args.train_ratio + args.val_ratio:
                    split = "val"
                else:
                    split = "test"

                cur_x, cur_y = sample_tile_center(map_size, tile_size)

                # TAR is not used for CUR-localization; set it equal to CUR for compatibility.
                tar_x, tar_y = cur_x, cur_y
                dx = 0.0
                dy = 0.0

                records.append(
                    {
                        "sample_id": sample_id,
                        "map_id": map_id,
                        "map_crop_x": int(crop_x),
                        "map_crop_y": int(crop_y),
                        "cur_tile_x": float(cur_x),
                        "cur_tile_y": float(cur_y),
                        "tar_tile_x": float(tar_x),
                        "tar_tile_y": float(tar_y),
                        "vector_dx": float(dx),
                        "vector_dy": float(dy),
                        "split": split,
                        "scale": float(scale),
                        "source_image": str(big_path.name),
                    }
                )
                sample_id += 1

    if not records:
        raise RuntimeError("No samples were generated. Check scales/map-size and the big image size.")

    df = pd.DataFrame.from_records(records)
    df.to_csv(out_csv, index=False)

    root = out_csv.parent
    df[df["split"] == "train"].to_csv(root / f"{out_csv.stem}_train.csv", index=False)
    df[df["split"] == "val"].to_csv(root / f"{out_csv.stem}_val.csv", index=False)
    df[df["split"] == "test"].to_csv(root / f"{out_csv.stem}_test.csv", index=False)

    print(f"Saved: {out_csv}  (rows={len(df)})")
    print(f"Scaled maps saved in: {map_dir}")


if __name__ == "__main__":
    main()

