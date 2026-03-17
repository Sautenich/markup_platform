import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image


RNG = np.random.default_rng()


def sample_lower_crop_origin(
    img_w: int,
    img_h: int,
    crop_size: int,
    lower_fraction: float = 0.5,
) -> Tuple[int, int]:
    """
    Sample the top-left corner (x, y) of a square crop of size `crop_size`
    restricted to the LOWER part of the image.

    lower_fraction=0.5 => lower half of the image is used.
    """
    if img_w < crop_size or img_h < crop_size:
        raise ValueError("Image is smaller than the requested crop size.")

    max_x = img_w - crop_size

    # Lower band [img_h * lower_fraction, img_h]
    band_start = int(img_h * lower_fraction)
    band_end = img_h - crop_size
    if band_end < band_start:
        # If the band is too small, just clamp to the last possible crop.
        band_start = max(0, img_h - crop_size)
        band_end = band_start

    x = int(RNG.integers(0, max_x + 1))
    y = int(RNG.integers(band_start, band_end + 1))
    return x, y


def sample_tile_center(map_size: int, tile_size: int) -> Tuple[int, int]:
    """
    Sample a tile center (cx, cy) fully inside a square crop of size `map_size`.
    """
    half = tile_size // 2
    low = half
    high = map_size - half
    cx = int(RNG.integers(low, high + 1))
    cy = int(RNG.integers(low, high + 1))
    return cx, cy


def generate_lower_part_annotations(
    input_image: Path,
    output_csv: Path,
    map_size: int = 2000,
    tile_size: int = 200,
    crops_per_image: int = 50,
    tiles_per_crop: int = 50,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    lower_fraction: float = 0.5,
) -> None:
    """
    Generate a dataset from the LOWER part of a big image.

    For each sampled crop (map_size x map_size) from the lower part,
    we sample multiple 200x200 tile centers inside that crop.

    The resulting CSV is similar to annotations.csv, but:
      - only ONE tile per row (TAR), no CUR/TAR pair
      - columns: sample_id, map_id, map_crop_x, map_crop_y, tar_tile_x, tar_tile_y, split
    """
    input_image = Path(input_image)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not input_image.exists():
        raise FileNotFoundError(f"Input image not found: {input_image}")

    Image.MAX_IMAGE_PIXELS = None
    big_img = Image.open(input_image).convert("RGB")
    img_w, img_h = big_img.size

    records = []
    sample_id = 0
    map_id = input_image.stem

    for _ in range(crops_per_image):
        try:
            crop_x, crop_y = sample_lower_crop_origin(
                img_w=img_w,
                img_h=img_h,
                crop_size=map_size,
                lower_fraction=lower_fraction,
            )
        except ValueError:
            break

        for _ in range(tiles_per_crop):
            tar_x, tar_y = sample_tile_center(map_size, tile_size)

            # assign split
            r = RNG.random()
            if r < train_ratio:
                split = "train"
            elif r < train_ratio + val_ratio:
                split = "val"
            else:
                split = "test"

            records.append(
                {
                    "sample_id": sample_id,
                    "map_id": map_id,
                    "map_crop_x": int(crop_x),
                    "map_crop_y": int(crop_y),
                    "tar_tile_x": float(tar_x),
                    "tar_tile_y": float(tar_y),
                    "split": split,
                }
            )
            sample_id += 1

    if not records:
        raise RuntimeError("No samples were generated. Check image size and parameters.")

    df = pd.DataFrame.from_records(records)
    df.to_csv(output_csv, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a dataset from the LOWER part of big_sample5_17.tif.\n"
            "Random 2000x2000 regions are sampled from the lower part of the image, "
            "then random 200x200 tiles are sampled inside each region. "
            "The output CSV is similar to annotations.csv but contains only one TAR tile per row."
        )
    )
    parser.add_argument(
        "--input-image",
        type=str,
        default="big_sample5_17.tif",
        help="Path to the big input image.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="dataset/annotations_lower_tar.csv",
        help="Where to save the generated annotations CSV.",
    )
    parser.add_argument(
        "--map-size",
        type=int,
        default=2000,
        help="Size of the large crop (e.g., 2000x2000).",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=200,
        help="Size of tile inside the crop (e.g., 200x200).",
    )
    parser.add_argument(
        "--crops-per-image",
        type=int,
        default=50,
        help="How many 2000x2000 crops to sample from the lower part of the image.",
    )
    parser.add_argument(
        "--tiles-per-crop",
        type=int,
        default=50,
        help="How many 200x200 tiles to sample inside each 2000x2000 crop.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--lower-fraction",
        type=float,
        default=0.5,
        help="Fraction of image height representing the LOWER band (0.5 => bottom half).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    generate_lower_part_annotations(
        input_image=Path(args.input_image),
        output_csv=Path(args.output_csv),
        map_size=args.map_size,
        tile_size=args.tile_size,
        crops_per_image=args.crops_per_image,
        tiles_per_crop=args.tiles_per_crop,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        lower_fraction=args.lower_fraction,
    )


if __name__ == "__main__":
    main()


