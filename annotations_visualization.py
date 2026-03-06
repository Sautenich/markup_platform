import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize annotated current/target tiles and ground-truth vector on satellite crop."
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="dataset/annotations.csv",
        help="Path to annotations CSV.",
    )
    parser.add_argument(
        "--map-dir",
        type=str,
        default="dataset/satellite_maps",
        help="Directory with original satellite images.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=-1,
        help="Row index in filtered annotations to visualize. If -1, choose random.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split to visualize: train | val | test | all.",
    )
    parser.add_argument(
        "--map-size",
        type=int,
        default=1000,
        help="Crop size (should match map_size used in dataset generation).",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=200,
        help="Tile size in pixels (should match tile_size used in dataset generation).",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="viz_annotations.png",
        help="Where to save the visualization image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    annotations_path = Path(args.annotations)
    map_dir = Path(args.map_dir)

    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")

    df = pd.read_csv(annotations_path)

    if args.split != "all" and "split" in df.columns:
        df = df[df["split"] == args.split].reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError("No rows left after filtering by split. Check annotations and --split.")

    if args.index < 0 or args.index >= len(df):
        idx = np.random.randint(0, len(df))
    else:
        idx = args.index

    row = df.iloc[idx]

    map_id = str(row["map_id"])
    crop_x = int(row["map_crop_x"])
    crop_y = int(row["map_crop_y"])

    cur_x = float(row["cur_tile_x"])
    cur_y = float(row["cur_tile_y"])
    tar_x = float(row["tar_tile_x"])
    tar_y = float(row["tar_tile_y"])

    dx = float(row["vector_dx"])
    dy = float(row["vector_dy"])

    # Load full satellite image
    map_path = map_dir / f"{map_id}.tif"
    if not map_path.exists():
        # try other common extensions
        for ext in (".jpg", ".jpeg", ".png", ".tiff"):
            alt = map_dir / f"{map_id}{ext}"
            if alt.exists():
                map_path = alt
                break
    if not map_path.exists():
        raise FileNotFoundError(f"Could not find image for map_id={map_id} in {map_dir}")

    full_img = Image.open(map_path).convert("RGB")

    # Crop 1000x1000 region used for this sample
    map_size = args.map_size
    crop = full_img.crop(
        (crop_x, crop_y, crop_x + map_size, crop_y + map_size)
    )  # (left, upper, right, lower)

    # Convert to OpenCV BGR image
    img_np = np.array(crop)  # RGB, uint8
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    tile_size = args.tile_size

    # Draw current tile rectangle (green)
    cur_top_left = (
        int(cur_x - tile_size / 2),
        int(cur_y - tile_size / 2),
    )
    cur_bottom_right = (
        int(cur_x + tile_size / 2),
        int(cur_y + tile_size / 2),
    )
    cv2.rectangle(img_bgr, cur_top_left, cur_bottom_right, color=(0, 255, 0), thickness=2)

    # Draw target tile rectangle (yellow)
    tar_top_left = (
        int(tar_x - tile_size / 2),
        int(tar_y - tile_size / 2),
    )
    tar_bottom_right = (
        int(tar_x + tile_size / 2),
        int(tar_y + tile_size / 2),
    )
    cv2.rectangle(img_bgr, tar_top_left, tar_bottom_right, color=(0, 255, 255), thickness=2)

    # Draw ground-truth vector (dx, dy) from current to target (red arrow)
    end_point = (
        int(cur_x + dx),
        int(cur_y + dy),
    )
    cv2.arrowedLine(
        img_bgr,
        (int(cur_x), int(cur_y)),
        end_point,
        color=(0, 0, 255),
        thickness=2,
        tipLength=0.05,
    )

    # Labels
    cv2.putText(
        img_bgr,
        "CUR",
        (int(cur_x) + 5, int(cur_y) - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img_bgr,
        "TAR",
        (int(tar_x) + 5, int(tar_y) - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        img_bgr,
        f"idx={idx}, split={row.get('split', 'NA')}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    out_path = Path(args.save_path)
    cv2.imwrite(str(out_path), img_bgr)
    print(f"Saved annotated visualization to {out_path}")


if __name__ == "__main__":
    main()

