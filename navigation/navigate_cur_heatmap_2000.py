import sys
from pathlib import Path as _Path

# Allow running as a script: `python navigation/navigate_cur_heatmap_2000.py`
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
import torchvision.transforms as T

from training_and_NN.cur_localization_net import CurLocalizationNet, build_localization_transforms


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Interactive navigation for CurLocalizationNet on 2000x2000 map tiles.\n"
            "Reads a CSV (annotations_big_cur_2000.csv) and a map-dir containing 2000x2000 images.\n"
            "Lets you move a 200x200 window and switch samples with N/P."
        )
    )
    p.add_argument(
        "--annotations",
        type=str,
        default="dataset/annotations_big_cur_2000.csv",
        help="CSV with columns: map_id, cur_tile_x/y, split (optional).",
    )
    p.add_argument(
        "--map-dir",
        type=str,
        default="dataset/satellite_maps/2000x2000test_big_sample",
        help="Directory containing 2000x2000 map images.",
    )
    p.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"])
    p.add_argument(
        "--map-id-prefix",
        type=str,
        default="",
        help="Optional filter: only keep samples whose map_id starts with this prefix.",
    )
    p.add_argument("--index", type=int, default=0, help="Index within the filtered subset.")
    p.add_argument("--map-size", type=int, default=2000, help="Map image size (pixels).")
    p.add_argument("--tile-size", type=int, default=200, help="Tile window size (pixels).")
    p.add_argument("--step", type=int, default=200, help="Move step in pixels.")
    p.add_argument(
        "--model-path",
        type=str,
        default="weights/cur_localization_net.pt",
        help="Path to trained CurLocalizationNet weights (optional).",
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--window-size", type=int, default=1200)
    return p.parse_args()


def _normalize_for_display(prob_map: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(prob_map, [1.0, 99.5])
    return np.clip((prob_map - lo) / max(hi - lo, 1e-12), 0.0, 1.0)


def _load_map_image(map_dir: Path, map_id: str) -> Image.Image:
    # Try common extensions
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        p = map_dir / f"{map_id}{ext}"
        if p.exists():
            Image.MAX_IMAGE_PIXELS = None
            return Image.open(p).convert("RGB")
    raise FileNotFoundError(f"Map image not found for map_id={map_id} in {map_dir}")


def main() -> None:
    args = parse_args()
    ann_path = Path(args.annotations)
    map_dir = Path(args.map_dir)
    device = torch.device(args.device)

    if not ann_path.exists():
        raise FileNotFoundError(f"Annotations CSV not found: {ann_path}")
    if not map_dir.exists():
        raise FileNotFoundError(f"Map dir not found: {map_dir}")

    df = pd.read_csv(ann_path)
    if df.empty:
        raise RuntimeError(f"No rows in annotations CSV: {ann_path}")

    if args.split != "all" and "split" in df.columns:
        df = df[df["split"] == args.split].reset_index(drop=True)

    if args.map_id_prefix:
        prefixes = tuple(p.strip() for p in args.map_id_prefix.split(",") if p.strip())
        if prefixes:
            df = df[df["map_id"].astype(str).str.startswith(prefixes)].reset_index(drop=True)

    if df.empty:
        raise RuntimeError("No samples matched filters.")

    map_size = int(args.map_size)
    tile_size = int(args.tile_size)
    half_tile = tile_size // 2
    step = int(args.step)

    to_tensor = T.ToTensor()
    transform_pair = build_localization_transforms()

    # Model
    model = CurLocalizationNet().to(device)
    model_path = Path(args.model_path)
    if model_path.exists():
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded model weights from {model_path}")
    else:
        print(f"Model weights not found at {model_path}, using randomly initialized model.")
    model.eval()

    cv2.namedWindow("Navigation CUR Heatmap (2000)", cv2.WINDOW_NORMAL)
    print(
        "Controls:\n"
        "  W/S or Up/Down    - move tile window vertically\n"
        "  A/D or Left/Right - move tile window horizontally\n"
        "  R                 - reset window to GT CUR\n"
        "  N                 - next sample\n"
        "  P                 - previous sample\n"
        "  Q or Esc          - quit\n"
        "Tip: click the OpenCV window to focus it.\n"
    )

    idx = 0
    pos = 0
    base_crop = None  # type: ignore[assignment]
    base_sat_in = None  # type: ignore[assignment]
    gt_x = gt_y = 0.0
    cur_x = cur_y = 0.0

    def load_sample(new_pos: int):
        nonlocal pos, idx, base_crop, base_sat_in, gt_x, gt_y, cur_x, cur_y
        pos = int(new_pos) % len(df)
        row = df.iloc[pos]
        map_id = str(row["map_id"])
        img = _load_map_image(map_dir, map_id)

        # Ensure size matches expected (or resize)
        if img.size != (map_size, map_size):
            img = img.resize((map_size, map_size), resample=Image.BICUBIC)

        base_crop = img
        gt_x = float(row["cur_tile_x"])
        gt_y = float(row["cur_tile_y"])
        cur_x, cur_y = gt_x, gt_y

        # Precompute normalized map tensor once per sample
        # (dummy tile just to reuse the same transform pipeline signature)
        dummy_tile = base_crop.crop((0, 0, tile_size, tile_size))
        sat_in, _ = transform_pair(base_crop, dummy_tile)
        base_sat_in = sat_in.unsqueeze(0).to(device)

    load_sample(int(args.index))

    while True:
        cur_x = float(np.clip(cur_x, half_tile, map_size - half_tile))
        cur_y = float(np.clip(cur_y, half_tile, map_size - half_tile))

        tile = base_crop.crop(
            (
                int(cur_x - half_tile),
                int(cur_y - half_tile),
                int(cur_x + half_tile),
                int(cur_y + half_tile),
            )
        )
        _, tile_in = transform_pair(base_crop, tile)
        tile_in = tile_in.unsqueeze(0).to(device)

        with torch.inference_mode():
            logits = model(base_sat_in, tile_in)[0, 0]
        heatmap = torch.softmax(logits.view(-1), dim=0).view_as(logits)
        heatmap_np = heatmap.cpu().numpy()

        heatmap_vis = _normalize_for_display(heatmap_np)
        heatmap_resized = cv2.resize(heatmap_vis, (map_size, map_size), interpolation=cv2.INTER_CUBIC)

        img_np = np.array(base_crop)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        heatmap_color = cv2.applyColorMap((heatmap_resized * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_bgr, 0.5, heatmap_color, 0.5, 0)

        # GT box
        gt_tl = (int(gt_x - half_tile), int(gt_y - half_tile))
        gt_br = (int(gt_x + half_tile), int(gt_y + half_tile))
        cv2.rectangle(overlay, gt_tl, gt_br, (0, 255, 0), 2)
        cv2.putText(overlay, "GT", (gt_tl[0] + 5, gt_tl[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # CUR window
        cur_tl = (int(cur_x - half_tile), int(cur_y - half_tile))
        cur_br = (int(cur_x + half_tile), int(cur_y + half_tile))
        cv2.rectangle(overlay, cur_tl, cur_br, (255, 255, 255), 2)
        cv2.putText(overlay, "CUR", (cur_tl[0] + 5, cur_tl[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # peak dot
        hf, wf = heatmap_np.shape
        peak_idx = int(heatmap_np.reshape(-1).argmax())
        peak_y = peak_idx // wf
        peak_x = peak_idx % wf
        peak_x_px = int(round(peak_x * (map_size - 1) / max(wf - 1, 1)))
        peak_y_px = int(round(peak_y * (map_size - 1) / max(hf - 1, 1)))
        cv2.circle(overlay, (peak_x_px, peak_y_px), 6, (0, 255, 255), -1)

        map_id = str(df.iloc[pos]["map_id"])
        status = f"pos {pos}/{len(df)-1} | map_id {map_id} | center ({cur_x:.0f},{cur_y:.0f}) | peak ({peak_x_px},{peak_y_px})"
        cv2.putText(overlay, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        if max(overlay.shape[:2]) > args.window_size:
            scale = args.window_size / max(overlay.shape[:2])
            disp = cv2.resize(overlay, (int(overlay.shape[1] * scale), int(overlay.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        else:
            disp = overlay

        cv2.imshow("Navigation CUR Heatmap (2000)", disp)
        key = cv2.waitKeyEx(0)

        KEY_UP = 2490368
        KEY_DOWN = 2621440
        KEY_LEFT = 2424832
        KEY_RIGHT = 2555904

        if key in (ord("q"), ord("Q"), 27):
            break
        elif key in (ord("w"), ord("W"), KEY_UP, 82):
            cur_y -= step
        elif key in (ord("s"), ord("S"), KEY_DOWN, 84):
            cur_y += step
        elif key in (ord("a"), ord("A"), KEY_LEFT, 81):
            cur_x -= step
        elif key in (ord("d"), ord("D"), KEY_RIGHT, 83):
            cur_x += step
        elif key in (ord("r"), ord("R")):
            cur_x, cur_y = gt_x, gt_y
        elif key in (ord("n"), ord("N")):
            load_sample(pos + 1)
        elif key in (ord("p"), ord("P")):
            load_sample(pos - 1)

    cv2.destroyWindow("Navigation CUR Heatmap (2000)")


if __name__ == "__main__":
    main()

