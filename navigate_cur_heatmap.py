import argparse
from pathlib import Path

import cv2
import numpy as np

import torch
import torchvision.transforms as T

from create_dataset import NavigationDataset
from cur_localization_net import CurLocalizationNet, build_localization_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive navigation over a NavigationDataset crop for CurLocalizationNet.\n"
            "Fixes a map crop from annotations.csv and lets you move a 200x200 tile window.\n"
            "At each step it runs CurLocalizationNet(map_crop, current_tile) and shows the heatmap.\n"
        )
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="dataset/annotations.csv",
        help="Path to dataset/annotations.csv used by NavigationDataset.",
    )
    parser.add_argument(
        "--map-dir",
        type=str,
        default="dataset/satellite_maps",
        help="Directory with satellite maps referenced by annotations.csv.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "all"],
        help="Which split to sample from. Use 'all' to use the entire CSV.",
    )
    parser.add_argument(
        "--map-id-prefix",
        type=str,
        default="",
        help="Optional filter: only keep samples whose map_id starts with this prefix (e.g. 'big_sample5_17').",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Optional filter: only keep samples with this scale value (requires 'scale' column in CSV).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="cur_localization_net.pt",
        help="Path to trained CurLocalizationNet weights (optional).",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Row index inside the chosen split (0-based).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=200,
        help="Step in pixels when moving the tile window.",
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


def _normalize_for_display(prob_map: np.ndarray) -> np.ndarray:
    # prob_map is already in [0,1] (softmax), but dynamic range is tiny.
    # Percentile clipping gives a more readable visualization than min-max.
    lo, hi = np.percentile(prob_map, [1.0, 99.5])
    return np.clip((prob_map - lo) / max(hi - lo, 1e-12), 0.0, 1.0)


def main() -> None:
    args = parse_args()

    ann_path = Path(args.annotations)
    map_dir = Path(args.map_dir)
    model_path = Path(args.model_path)
    device = torch.device(args.device)

    if not ann_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {ann_path}")
    if not map_dir.exists():
        raise FileNotFoundError(f"Map dir not found: {map_dir}")

    # No transform here: we want the raw crop to crop tiles interactively.
    ds = NavigationDataset(
        csv_file=str(ann_path),
        map_dir=str(map_dir),
        split=None if args.split == "all" else args.split,
        transform=None,
    )
    if len(ds) == 0:
        raise RuntimeError(f"Split '{args.split}' is empty. Check the 'split' column in {ann_path}.")

    # Build a filtered index list (so N/P navigates within the desired subset).
    indices = list(range(len(ds)))
    if args.map_id_prefix:
        prefixes = tuple(p.strip() for p in args.map_id_prefix.split(",") if p.strip())
        if prefixes:
            indices = [
                i
                for i in indices
                if str(ds.annotations.iloc[i]["map_id"]).startswith(prefixes)  # type: ignore[attr-defined]
            ]

    if args.scale is not None and "scale" in ds.annotations.columns:  # type: ignore[attr-defined]
        target_scale = float(args.scale)
        indices = [
            i
            for i in indices
            if abs(float(ds.annotations.iloc[i]["scale"]) - target_scale) < 1e-9  # type: ignore[attr-defined]
        ]

    if not indices:
        raise RuntimeError("No samples matched filters (--map-id-prefix/--scale).")

    map_size = int(ds.map_size)
    tile_size = int(ds.tile_size)
    half_tile = tile_size // 2
    step = int(args.step)

    to_pil = T.ToPILImage()

    # Model + transforms
    transform_pair = build_localization_transforms()

    def load_sample(new_idx: int):
        nonlocal idx, base_crop, base_sat_in, gt_x, gt_y, cur_x, cur_y, pos_in_subset

        # new_idx is position inside 'indices' (wrap-around navigation)
        if len(indices) == 0:
            raise RuntimeError("No indices to load.")
        pos_in_subset = int(new_idx) % len(indices)
        idx = indices[pos_in_subset]
        sat_img, _cur_tile_gt, _tar_tile, labels = ds[idx]

        # Base crop as PIL (for easy tile cropping)
        base_crop = to_pil(sat_img)  # (map_size, map_size)

        # GT CUR center inside crop (pixels)
        gt_cur_xy = labels["cur_xy"] * map_size
        gt_x = float(gt_cur_xy[0].item())
        gt_y = float(gt_cur_xy[1].item())

        # Start/reset agent window at GT
        cur_x = gt_x
        cur_y = gt_y

        # Precompute normalized map tensor once per sample
        base_sat_in_local, _ = transform_pair(base_crop, base_crop.crop((0, 0, tile_size, tile_size)))
        base_sat_in = base_sat_in_local.unsqueeze(0).to(device)

    # init first sample
    idx = 0
    pos_in_subset = 0
    base_crop = None  # type: ignore[assignment]
    base_sat_in = None  # type: ignore[assignment]
    gt_x = gt_y = 0.0
    cur_x = cur_y = 0.0
    load_sample(int(args.index))

    model = CurLocalizationNet().to(device)
    if model_path.exists():
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded model weights from {model_path}")
    else:
        print(f"Model weights not found at {model_path}, using randomly initialized model.")
    model.eval()

    cv2.namedWindow("Navigation CUR Heatmap", cv2.WINDOW_NORMAL)

    print(
        "Controls:\n"
        "  W/S or Up/Down    - move tile window vertically inside crop\n"
        "  A/D or Left/Right - move tile window horizontally inside crop\n"
        "  R                 - reset tile window to GT CUR position\n"
        "  N                 - next tile/sample\n"
        "  P                 - previous tile/sample\n"
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

        # Prepare tile tensor
        _, tile_in = transform_pair(base_crop, tile)
        tile_in = tile_in.unsqueeze(0).to(device)

        with torch.inference_mode():
            logits = model(base_sat_in, tile_in)[0, 0]  # (Hf, Wf)

        # Convert logits to heatmap probabilities
        heatmap = torch.softmax(logits.view(-1), dim=0).view_as(logits)
        heatmap_np = heatmap.cpu().numpy()

        # Resize heatmap to crop size
        h_map, w_map = map_size, map_size
        heatmap_vis = _normalize_for_display(heatmap_np)
        heatmap_resized = cv2.resize(
            heatmap_vis,
            (w_map, h_map),
            interpolation=cv2.INTER_CUBIC,
        )

        # Base image in BGR
        crop_np = np.array(base_crop)
        img_bgr = cv2.cvtColor(crop_np, cv2.COLOR_RGB2BGR)

        # Color heatmap
        heatmap_color = cv2.applyColorMap(
            (heatmap_resized * 255.0).astype(np.uint8),
            cv2.COLORMAP_JET,
        )

        # Blend
        alpha = 0.5
        overlay = cv2.addWeighted(img_bgr, 1.0 - alpha, heatmap_color, alpha, 0)

        # Draw GT CUR position (fixed inside crop)
        gt_tl = (int(gt_x - half_tile), int(gt_y - half_tile))
        gt_br = (int(gt_x + half_tile), int(gt_y + half_tile))
        cv2.rectangle(overlay, gt_tl, gt_br, (0, 255, 0), 2)
        cv2.putText(
            overlay,
            "GT",
            (gt_tl[0] + 5, gt_tl[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        # Draw current tile window (agent view)
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

        # Show model peak (argmax) in crop coordinates
        hf, wf = heatmap_np.shape
        peak_idx = int(heatmap_np.reshape(-1).argmax())
        peak_y = peak_idx // wf
        peak_x = peak_idx % wf
        peak_x_px = int(round(peak_x * (w_map - 1) / max(wf - 1, 1)))
        peak_y_px = int(round(peak_y * (h_map - 1) / max(hf - 1, 1)))
        cv2.circle(overlay, (peak_x_px, peak_y_px), 6, (0, 255, 255), -1)

        # Status text
        map_id = str(ds.annotations.iloc[idx]["map_id"])  # type: ignore[attr-defined]
        status = (
            f"split {args.split} | sample {pos_in_subset}/{len(indices)-1} | "
            f"map_id {map_id} | "
            f"tile center ({cur_x:.1f}, {cur_y:.1f}) | "
            f"peak ({peak_x_px}, {peak_y_px})"
        )
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

        cv2.imshow("Navigation CUR Heatmap", disp)
        # waitKeyEx is required for arrow keys on many platforms (extended codes).
        key = cv2.waitKeyEx(0)

        # Arrow keys (common OpenCV codes)
        KEY_UP = 2490368
        KEY_DOWN = 2621440
        KEY_LEFT = 2424832
        KEY_RIGHT = 2555904

        if key in (ord("q"), ord("Q"), 27):
            break
        elif key in (ord("w"), ord("W"), KEY_UP, 82):  # Up
            cur_y -= step
        elif key in (ord("s"), ord("S"), KEY_DOWN, 84):  # Down
            cur_y += step
        elif key in (ord("a"), ord("A"), KEY_LEFT, 81):  # Left
            cur_x -= step
        elif key in (ord("d"), ord("D"), KEY_RIGHT, 83):  # Right
            cur_x += step
        elif key in (ord("r"), ord("R")):
            cur_x = gt_x
            cur_y = gt_y
        elif key in (ord("n"), ord("N")):
            load_sample(pos_in_subset + 1)
        elif key in (ord("p"), ord("P")):
            load_sample(pos_in_subset - 1)

    cv2.destroyWindow("Navigation CUR Heatmap")


if __name__ == "__main__":
    main()

