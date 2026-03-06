import argparse
from pathlib import Path
import shutil

from create_dataset import generate_annotations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a tiny synthetic dataset based on an RGB gradient image. "
            "Useful for checking that the whole training/eval pipeline works."
        )
    )
    parser.add_argument(
        "--rgb-image",
        type=str,
        default="rgb_gradient.png",
        help="Path to the RGB gradient image (default: rgb_gradient.png in project root).",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="dataset_rgb",
        help="Root directory for the synthetic dataset (will be created).",
    )
    parser.add_argument(
        "--samples-per-image",
        type=int,
        default=1000,
        help="How many samples to generate from the RGB image.",
    )
    parser.add_argument(
        "--map-size",
        type=int,
        default=1000,
        help="Square map size in pixels. Should not exceed the RGB image size.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=200,
        help="Tile size in pixels.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rgb_path = Path(args.rgb_image)
    if not rgb_path.exists():
        raise FileNotFoundError(
            f"RGB image not found at {rgb_path}. "
            f"Run test_rgb_gradient.py first or specify --rgb-image."
        )

    root = Path(args.root)
    sat_dir = root / "satellite_maps"
    sat_dir.mkdir(parents=True, exist_ok=True)

    # Copy the RGB image into the satellite_maps directory so that
    # generate_annotations can treat it as a normal map.
    dst_img = sat_dir / "rgb_map.png"
    shutil.copy2(rgb_path, dst_img)

    annotations_csv = root / "annotations.csv"

    generate_annotations(
        satellite_dir=sat_dir,
        output_csv=annotations_csv,
        map_size=args.map_size,
        tile_size=args.tile_size,
        samples_per_image=args.samples_per_image,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    print(
        f"Synthetic RGB dataset created under {root}.\n"
        f"Annotations: {annotations_csv}\n"
        f"Maps dir: {sat_dir}"
    )


if __name__ == "__main__":
    main()


