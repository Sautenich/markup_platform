import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize 1000x1000 crops defined in a manual_interest annotations CSV.\n"
            "Use R to go forward, E to go back, Q to quit."
        )
    )
    parser.add_argument(
        "--input-image",
        type=str,
        default="big_sample5_17.tif",
        help="Path to the large source image (e.g. big_sample5_17.tif).",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="dataset/annotations_manual_interest.csv",
        help="Path to CSV created by create_manual_interest_dataset.py.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=1000,
        help="Maximum display window size (longest side in pixels).",
    )
    parser.add_argument(
        "--map-size",
        type=int,
        default=2000,
        help="Map crop size in pixels (must match map_size used in annotations).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start visualization from this row index in CSV.",
    )
    return parser.parse_args()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def main() -> None:
    args = parse_args()

    img_path = Path(args.input_image)
    ann_path = Path(args.annotations)

    if not img_path.exists():
        raise FileNotFoundError(f"Input image not found: {img_path}")
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotations CSV not found: {ann_path}")

    # Big image can be huge
    Image.MAX_IMAGE_PIXELS = None
    big_img = Image.open(img_path).convert("RGB")
    img_w, img_h = big_img.size

    df = pd.read_csv(ann_path)
    if df.empty:
        raise RuntimeError(f"No rows in annotations CSV: {ann_path}")

    # Only use manual_interest rows if present
    if "split" in df.columns:
        mask = df["split"] == "manual_interest"
        if mask.any():
            df = df[mask].reset_index(drop=True)

    n = len(df)
    if n == 0:
        raise RuntimeError("No rows with split == 'manual_interest' found.")

    idx = clamp(args.start_index, 0, n - 1)

    print(
        f"Loaded {n} samples from {ann_path}\n"
        f"Controls in window:\n"
        f"  R - next sample\n"
        f"  E - previous sample\n"
        f"  Q - quit\n"
    )

    cv2.namedWindow("Manual Interest Crops", cv2.WINDOW_NORMAL)

    while True:
        row = df.iloc[int(idx)]

        x0 = int(row["map_crop_x"])
        y0 = int(row["map_crop_y"])
        map_size = int(args.map_size)

        x1 = x0 + map_size
        y1 = y0 + map_size

        # Clamp just in case
        x0 = int(clamp(x0, 0, img_w - map_size))
        y0 = int(clamp(y0, 0, img_h - map_size))
        x1 = x0 + map_size
        y1 = y0 + map_size

        crop = big_img.crop((x0, y0, x1, y1))
        crop_np = np.array(crop)
        img_bgr = cv2.cvtColor(crop_np, cv2.COLOR_RGB2BGR)

        # Draw current and target tiles if present
        if all(k in row for k in ("cur_tile_x", "cur_tile_y", "tar_tile_x", "tar_tile_y")):
            tile_size = 200
            half_tile = tile_size // 2

            cur_x = float(row["cur_tile_x"])
            cur_y = float(row["cur_tile_y"])
            tar_x = float(row["tar_tile_x"])
            tar_y = float(row["tar_tile_y"])

            cur_tl = (int(cur_x - half_tile), int(cur_y - half_tile))
            cur_br = (int(cur_x + half_tile), int(cur_y + half_tile))
            tar_tl = (int(tar_x - half_tile), int(tar_y - half_tile))
            tar_br = (int(tar_x + half_tile), int(tar_y + half_tile))

            cv2.rectangle(img_bgr, cur_tl, cur_br, (0, 255, 0), 2)
            cv2.putText(
                img_bgr,
                "CUR",
                (cur_tl[0] + 5, cur_tl[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.rectangle(img_bgr, tar_tl, tar_br, (0, 255, 255), 2)
            cv2.putText(
                img_bgr,
                "TAR",
                (tar_tl[0] + 5, tar_tl[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Draw geometric vector from CUR to TAR as an arrow
            arrow_start = (int(cur_x), int(cur_y))
            arrow_end = (int(tar_x), int(tar_y))
            cv2.arrowedLine(
                img_bgr,
                arrow_start,
                arrow_end,
                (0, 0, 255),
                2,
                tipLength=0.05,
            )

        # Status text: index and coords
        text = (
            f"idx {int(idx)}/{n-1} | crop ({x0},{y0})-({x1},{y1}) | "
            "R: next, E: prev, Q: quit"
        )
        cv2.putText(
            img_bgr,
            text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Resize for display if needed
        h, w = img_bgr.shape[:2]
        if max(h, w) > args.window_size:
            scale = args.window_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            disp = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            disp = img_bgr

        cv2.imshow("Manual Interest Crops", disp)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            break
        elif key in (ord("r"), ord("R")):
            idx = (idx + 1) % n
        elif key in (ord("e"), ord("E")):
            idx = (idx - 1 + n) % n

    cv2.destroyWindow("Manual Interest Crops")


if __name__ == "__main__":
    main()


