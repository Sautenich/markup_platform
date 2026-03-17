import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
from torchvision.transforms import ToTensor, ToPILImage

from tile_localization_net import TileLocalizationNet, build_tile_localization_transforms
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize TileLocalizationNet heatmap on TAR-only lower-part dataset "
            "(annotations_lower_tar.csv + big_sample5_17.tif)."
        )
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="dataset/annotations_lower_tar.csv",
        help="Path to TAR-only annotations CSV.",
    )
    parser.add_argument(
        "--big-image",
        type=str,
        default="big_sample5_17.tif",
        help="Path to the big image used to generate the CSV.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="tile_localization_lower_tar.pt",
        help="Path to trained TileLocalizationNet weights.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=-1,
        help="Row index in annotations CSV to visualize. If -1, choose random.",
    )
    parser.add_argument(
        "--map-size",
        type=int,
        default=2000,
        help="Size of the crop used when generating the dataset.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=200,
        help="Tile size (used only for drawing the GT box).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="viz_tile_lower_heatmap.png",
        help="Where to save the visualization image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ann_path = Path(args.annotations)
    big_image_path = Path(args.big_image)
    model_path = Path(args.model_path)
    device = torch.device(args.device)

    if not ann_path.exists():
        raise FileNotFoundError(f"Annotations CSV not found: {ann_path}")
    if not big_image_path.exists():
        raise FileNotFoundError(f"Big image not found: {big_image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    df = pd.read_csv(ann_path)
    if df.empty:
        raise RuntimeError(f"No rows in annotations CSV: {ann_path}")

    if args.index < 0 or args.index >= len(df):
        idx = np.random.randint(0, len(df))
    else:
        idx = args.index

    row = df.iloc[idx]

    # Load big image and crop region
    Image.MAX_IMAGE_PIXELS = None
    big_img = Image.open(big_image_path).convert("RGB")

    x0 = int(row["map_crop_x"])
    y0 = int(row["map_crop_y"])
    map_size = int(args.map_size)
    tile_size = int(args.tile_size)
    half = tile_size // 2

    crop = big_img.crop((x0, y0, x0 + map_size, y0 + map_size))

    tx = float(row["tar_tile_x"])
    ty = float(row["tar_tile_y"])

    tar_tile = crop.crop(
        (
            int(tx - half),
            int(ty - half),
            int(tx + half),
            int(ty + half),
        )
    )

    # Prepare tensors with same transforms as in training
    transform_pair = build_tile_localization_transforms()
    map_t, tile_t = transform_pair(crop, tar_tile)

    map_t = map_t.unsqueeze(0).to(device)
    tile_t = tile_t.unsqueeze(0).to(device)

    # Model
    model = TileLocalizationNet().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    with torch.inference_mode():
        logits = model(map_t, tile_t)[0, 0]  # (Hf, Wf)

    # Convert logits to probabilities for visualization
    heatmap = torch.softmax(logits.view(-1), dim=0).view_as(logits)
    heatmap_np = heatmap.cpu().numpy()

    # Resize heatmap to crop size
    h_map, w_map = crop.size[1], crop.size[0]
    h_f, w_f = heatmap_np.shape

    lo, hi = np.percentile(heatmap_np, [1.0, 99.5])
    heatmap_np = np.clip((heatmap_np - lo) / max(hi - lo, 1e-12), 0.0, 1.0)

    heatmap_resized = cv2.resize(
        heatmap_np,
        (w_map, h_map),
        interpolation=cv2.INTER_CUBIC,
    )

    # Base image (RGB -> BGR)
    to_tensor = ToTensor()
    to_pil = ToPILImage()
    crop_tensor = to_tensor(crop)
    img_np = crop_tensor.permute(1, 2, 0).numpy()
    img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Color heatmap
    heatmap_color = cv2.applyColorMap(
        (heatmap_resized * 255.0).astype(np.uint8),
        cv2.COLORMAP_JET,
    )

    # Blend
    alpha = 0.5
    overlay = cv2.addWeighted(img_bgr, 1.0 - alpha, heatmap_color, alpha, 0)

    # Draw GT TAR tile
    tar_tl = (int(tx - half), int(ty - half))
    tar_br = (int(tx + half), int(ty + half))
    cv2.rectangle(overlay, tar_tl, tar_br, (0, 255, 0), 2)
    cv2.putText(
        overlay,
        "TAR",
        (tar_tl[0] + 5, tar_tl[1] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

    cv2.imwrite(args.save_path, overlay)
    print(f"Saved tile localization heatmap to {args.save_path}")


if __name__ == "__main__":
    main()





