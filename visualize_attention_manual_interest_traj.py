import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch

from network_training_attention import AzimuthNetAttention, build_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize AzimuthNetAttention predictions on the "
            "manual_interest trajectory dataset (annotations_manual_interest_traj.csv).\n"
            "Uses a big source image (e.g., big_sample5_17.tif) and a trajectory CSV "
            "produced by create_manual_interest_trajectory.py. "
            "Controls: R - next, E - back, Q - quit."
        )
    )
    parser.add_argument(
        "--input-image",
        type=str,
        default="big_sample5_17.tif",
        help="Path to the large source image.",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="dataset/annotations_manual_interest_traj.csv",
        help="Path to trajectory CSV (with prev*_tile_x/y columns).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="azimuth_net_attention.pt",
        help="Path to trained AzimuthNetAttention weights.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=1200,
        help="Maximum display window size (longest side in pixels).",
    )
    parser.add_argument(
        "--map-size",
        type=int,
        default=2000,
        help="Map crop size in pixels (must match map_size used in annotations).",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=200,
        help="Tile size in pixels for CUR/TAR/prev boxes.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start visualization from this row index in CSV.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on.",
    )
    return parser.parse_args()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def main() -> None:
    args = parse_args()

    img_path = Path(args.input_image)
    ann_path = Path(args.annotations)
    model_path = Path(args.model_path)

    if not img_path.exists():
        raise FileNotFoundError(f"Input image not found: {img_path}")
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotations CSV not found: {ann_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    # Load big image
    Image.MAX_IMAGE_PIXELS = None
    big_img = Image.open(img_path).convert("RGB")
    img_w, img_h = big_img.size

    # Load annotations
    df = pd.read_csv(ann_path)
    if df.empty:
        raise RuntimeError(f"No rows in annotations CSV: {ann_path}")

    # Optional: filter to manual_interest split if column exists
    if "split" in df.columns:
        mask = df["split"] == "manual_interest"
        if mask.any():
            df = df[mask].reset_index(drop=True)

    # Previous tile columns (if present)
    prev_cols = [
        ("prev1_tile_x", "prev1_tile_y"),
        ("prev2_tile_x", "prev2_tile_y"),
        ("prev3_tile_x", "prev3_tile_y"),
        ("prev4_tile_x", "prev4_tile_y"),
    ]

    n = len(df)
    if n == 0:
        raise RuntimeError("No rows to visualize after optional split filtering.")

    idx = int(clamp(args.start_index, 0, n - 1))

    print(
        f"Loaded {n} trajectory samples from {ann_path}\n"
        "Controls in window:\n"
        "  R - next sample\n"
        "  E - previous sample\n"
        "  Q - quit\n"
    )

    # Prepare model
    device = torch.device(args.device)
    model = AzimuthNetAttention().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Transform matching training of AzimuthNetAttention
    transform_triplet = build_transforms(augment_tiles=False)

    cv2.namedWindow("Attention on Manual Interest Trajectory", cv2.WINDOW_NORMAL)

    map_size = int(args.map_size)
    tile_size = int(args.tile_size)
    half_tile = tile_size // 2

    while True:
        row = df.iloc[idx]

        # Map crop top-left corner
        x0 = int(row["map_crop_x"])
        y0 = int(row["map_crop_y"])

        x0 = int(clamp(x0, 0, img_w - map_size))
        y0 = int(clamp(y0, 0, img_h - map_size))
        x1 = x0 + map_size
        y1 = y0 + map_size

        # Crop big image
        crop = big_img.crop((x0, y0, x1, y1))

        # CUR and TAR centers (in crop coordinates)
        cur_x = float(row["cur_tile_x"])
        cur_y = float(row["cur_tile_y"])
        tar_x = float(row["tar_tile_x"])
        tar_y = float(row["tar_tile_y"])

        # Build PIL tiles for transform_triplet
        cur_tile = crop.crop(
            (
                int(cur_x - half_tile),
                int(cur_y - half_tile),
                int(cur_x + half_tile),
                int(cur_y + half_tile),
            )
        )
        tar_tile = crop.crop(
            (
                int(tar_x - half_tile),
                int(tar_y - half_tile),
                int(tar_x + half_tile),
                int(tar_y + half_tile),
            )
        )

        # Convert to tensors and normalize exactly as in training
        sat_t, cur_t, tar_t = transform_triplet(crop, cur_tile, tar_tile)

        with torch.inference_mode():
            sat_in = sat_t.unsqueeze(0).to(device)
            cur_in = cur_t.unsqueeze(0).to(device)
            tar_in = tar_t.unsqueeze(0).to(device)
            pred = model(sat_in, cur_in, tar_in)[0]  # (2,)

        cos_t = float(pred[0].item())
        sin_t = float(pred[1].item())

        # Convert crop to BGR for OpenCV visualization (use unnormalized crop)
        crop_np = np.array(crop)
        img_bgr = cv2.cvtColor(crop_np, cv2.COLOR_RGB2BGR)

        # Draw previous tiles if present
        colors_prev = [
            (255, 128, 0),   # orange
            (255, 0, 255),   # magenta
            (0, 255, 255),   # cyan
            (128, 128, 255), # light blue
        ]
        for i, (cx_col, cy_col) in enumerate(prev_cols):
            if cx_col in row and cy_col in row and not pd.isna(row[cx_col]) and not pd.isna(row[cy_col]):
                px = float(row[cx_col])
                py = float(row[cy_col])
                tl = (int(px - half_tile), int(py - half_tile))
                br = (int(px + half_tile), int(py + half_tile))
                cv2.rectangle(img_bgr, tl, br, colors_prev[i], 2)
                cv2.putText(
                    img_bgr,
                    f"P{i+1}",
                    (tl[0] + 5, tl[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    colors_prev[i],
                    1,
                    cv2.LINE_AA,
                )

        # Draw CUR and TAR tiles
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

        # Draw ground-truth geometric arrow from CUR to TAR (red)
        cv2.arrowedLine(
            img_bgr,
            (int(cur_x), int(cur_y)),
            (int(tar_x), int(tar_y)),
            (0, 0, 255),
            2,
            tipLength=0.05,
        )

        # Draw model-predicted direction arrow from CUR (blue)
        arrow_len = map_size * 0.4
        pred_dx = cos_t * arrow_len
        pred_dy = sin_t * arrow_len

        pred_end = (
            int(cur_x + pred_dx),
            int(cur_y + pred_dy),
        )

        cv2.arrowedLine(
            img_bgr,
            (int(cur_x), int(cur_y)),
            pred_end,
            (255, 0, 0),
            3,
            tipLength=0.05,
        )

        # Status text
        text = (
            f"idx {idx}/{n-1} | crop ({x0},{y0})-({x1},{y1}) | "
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

        # Resize for display
        h, w = img_bgr.shape[:2]
        if max(h, w) > args.window_size:
            scale = args.window_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            disp = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            disp = img_bgr

        cv2.imshow("Attention on Manual Interest Trajectory", disp)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            break
        elif key in (ord("r"), ord("R")):
            idx = (idx + 1) % n
        elif key in (ord("e"), ord("E")):
            idx = (idx - 1 + n) % n

    cv2.destroyWindow("Attention on Manual Interest Trajectory")


if __name__ == "__main__":
    main()


