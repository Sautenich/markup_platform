import sys
from pathlib import Path as _Path

# Allow running as a script: `python navigation/navigate_upper_heatmap.py`
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch

from training_and_NN.tile_localization_net import TileLocalizationNet, build_tile_localization_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive navigation simulator over the UPPER-part validation annotations.\n"
            "Uses annotations_val_upper_part.csv + big_sample5_17.tif and a trained "
            "TileLocalizationNet to show a heatmap while you move a window over the map."
        )
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="dataset/annotations_val_upper_part.csv",
        help="Path to upper-part annotations CSV (one TAR tile per row).",
    )
    parser.add_argument(
        "--big-image",
        type=str,
        default="big_sample5_17.tif",
        help="Path to the big image.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="weights/tile_localization_lower_tar2.pt",
        help="Path to trained TileLocalizationNet weights.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Row index in annotations CSV to use as the target tile (0-based).",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=200,
        help="Tile size in pixels (for drawing GT box).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=200,
        help="Step in pixels when moving the 200x200 tile window with keys.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for the model.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=1200,
        help="Max size of the display window (image will be downscaled to fit).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ann_path = Path(args.annotations)
    big_image_path = Path(args.big_image)
    model_path = Path(args.model_path)

    if not ann_path.exists():
        raise FileNotFoundError(f"Annotations CSV not found: {ann_path}")
    if not big_image_path.exists():
        raise FileNotFoundError(f"Big image not found: {big_image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    df = pd.read_csv(ann_path)
    if df.empty:
        raise RuntimeError(f"No rows in annotations CSV: {ann_path}")

    idx = int(np.clip(args.index, 0, len(df) - 1))
    row = df.iloc[idx]

    # Load big image
    Image.MAX_IMAGE_PIXELS = None
    big_img = Image.open(big_image_path).convert("RGB")
    img_w, img_h = big_img.size

    map_size = 2000  # fixed, as in annotations_val_upper_part
    tile_size = int(args.tile_size)
    half_tile = tile_size // 2
    step = int(args.step)

    # Ground-truth crop & TAR position (in that crop)
    base_x0 = int(row["map_crop_x"])
    base_y0 = int(row["map_crop_y"])
    tar_x_rel = float(row["tar_tile_x"])
    tar_y_rel = float(row["tar_tile_y"])

    # Base crop is fixed map input for the network
    base_crop = big_img.crop((base_x0, base_y0, base_x0 + map_size, base_y0 + map_size))
    # Current "view" window center inside the crop (starts at GT TAR position)
    cur_x = tar_x_rel
    cur_y = tar_y_rel

    # Prepare model + transforms
    device = torch.device(args.device)
    transform_pair = build_tile_localization_transforms()

    model = TileLocalizationNet().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    cv2.namedWindow("Navigation Upper Heatmap", cv2.WINDOW_NORMAL)

    print(
        "Controls:\n"
        "  W/S or Up/Down    - move 200x200 tile window vertically inside crop\n"
        "  A/D or Left/Right - move 200x200 tile window horizontally inside crop\n"
        "  R                 - reset tile window to GT TAR position\n"
        "  Q or Esc          - quit\n"
    )

    while True:
        # Clamp tile window center inside crop bounds
        cur_x = float(np.clip(cur_x, half_tile, map_size - half_tile))
        cur_y = float(np.clip(cur_y, half_tile, map_size - half_tile))

        # Current tile window inside the fixed crop
        tile = base_crop.crop(
            (
                int(cur_x - half_tile),
                int(cur_y - half_tile),
                int(cur_x + half_tile),
                int(cur_y + half_tile),
            )
        )

        crop = base_crop  # full 2000x2000 crop as map input

        # Prepare tensors
        map_t, tile_t = transform_pair(crop, tile)
        map_t = map_t.unsqueeze(0).to(device)
        tile_t = tile_t.unsqueeze(0).to(device)

        with torch.inference_mode():
            logits = model(map_t, tile_t)[0, 0]  # (Hf, Wf)

        # Convert logits to heatmap probabilities
        heatmap = torch.softmax(logits.view(-1), dim=0).view_as(logits)
        heatmap_np = heatmap.cpu().numpy()

        h_f, w_f = heatmap_np.shape
        h_map, w_map = crop.size[1], crop.size[0]

        # Normalize and resize
        heatmap_np = heatmap_np - heatmap_np.min()
        if heatmap_np.max() > 0:
            heatmap_np = heatmap_np / heatmap_np.max()

        heatmap_resized = cv2.resize(
            heatmap_np,
            (w_map, h_map),
            interpolation=cv2.INTER_CUBIC,
        )

        # Base image in BGR
        crop_np = np.array(crop)
        img_bgr = cv2.cvtColor(crop_np, cv2.COLOR_RGB2BGR)

        # Color heatmap
        heatmap_color = cv2.applyColorMap(
            (heatmap_resized * 255.0).astype(np.uint8),
            cv2.COLORMAP_JET,
        )

        # Blend
        alpha = 0.5
        overlay = cv2.addWeighted(img_bgr, 1.0 - alpha, heatmap_color, alpha, 0)

        # Draw GT TAR position (fixed inside crop)
        tar_tl = (int(tar_x_rel - half_tile), int(tar_y_rel - half_tile))
        tar_br = (int(tar_x_rel + half_tile), int(tar_y_rel + half_tile))
        cv2.rectangle(overlay, tar_tl, tar_br, (0, 255, 0), 2)
        cv2.putText(
            overlay,
            "GT",
            (tar_tl[0] + 5, tar_tl[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        # Draw current 200x200 tile window (agent view)
        cur_tl = (int(cur_x - half_tile), int(cur_y - half_tile))
        cur_br = (int(cur_x + half_tile), int(cur_y + half_tile))
        cv2.rectangle(overlay, cur_tl, cur_br, (255, 255, 255), 2)
        cv2.putText(
            overlay,
            "CUR",
            (cur_tl[0] + 5, cur_tl[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Status text
        status = f"idx {idx}/{len(df)-1} | tile center ({cur_x:.1f}, {cur_y:.1f})"
        cv2.putText(
            overlay,
            status,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Resize for display
        h_disp, w_disp = overlay.shape[:2]
        if max(h_disp, w_disp) > args.window_size:
            scale = args.window_size / max(h_disp, w_disp)
            new_w, new_h = int(w_disp * scale), int(h_disp * scale)
            disp = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            disp = overlay

        cv2.imshow("Navigation Upper Heatmap", disp)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            break
        elif key in (ord("w"), ord("W"), 82):  # Up
            cur_y -= step
        elif key in (ord("s"), ord("S"), 84):  # Down
            cur_y += step
        elif key in (ord("a"), ord("A"), 81):  # Left
            cur_x -= step
        elif key in (ord("d"), ord("D"), 83):  # Right
            cur_x += step
        elif key in (ord("r"), ord("R")):
            cur_x = tar_x_rel
            cur_y = tar_y_rel

    cv2.destroyWindow("Navigation Upper Heatmap")


if __name__ == "__main__":
    main()


