import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
import cv2
import numpy as np

from create_dataset import NavigationDataset
from network_training import AzimuthNet


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize azimuth prediction on an RGB-gradient based satellite crop."
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="dataset_rgb/annotations.csv",
        help="Path to annotations CSV.",
    )
    parser.add_argument(
        "--map-dir",
        type=str,
        default="dataset_rgb/satellite_maps",
        help="Directory with RGB-gradient satellite images.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="azimuth_net.pt",
        help="Path to trained model weights.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=-1,
        help="Sample index in test split to visualize. If -1, choose random.",
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
        default="viz_rgb_example.png",
        help="Where to save the visualization image.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    annotations = Path(args.annotations)
    map_dir = Path(args.map_dir)
    device = torch.device(args.device)

    # Dataset WITHOUT normalization so that we can visualize easily.
    dataset = NavigationDataset(
        csv_file=str(annotations),
        map_dir=str(map_dir),
        split="test",
        transform=None,
    )

    if len(dataset) == 0:
        raise RuntimeError("Test split is empty. Check annotations CSV and 'split' column.")

    if args.index < 0 or args.index >= len(dataset):
        idx = torch.randint(0, len(dataset), (1,)).item()
    else:
        idx = args.index

    sat_img, cur_tile, tar_tile, labels = dataset[idx]

    # sat_img, cur_tile, tar_tile are tensors in [0,1]
    # Prepare inputs for the model (with the same normalization as in training).
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    sat_in = normalize(sat_img.clone()).unsqueeze(0).to(device)
    cur_in = normalize(cur_tile.clone()).unsqueeze(0).to(device)
    tar_in = normalize(tar_tile.clone()).unsqueeze(0).to(device)

    # Load model
    model = AzimuthNet().to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    with torch.inference_mode():
        pred = model(sat_in, cur_in, tar_in)[0]  # (2,)

    # Predicted direction as (cos, sin)
    cos_t, sin_t = pred[0].item(), pred[1].item()

    # Ground truth centers (in normalized coordinates in [0,1]) -> pixels
    map_size = dataset.map_size
    tile_size = dataset.tile_size

    cur_xy = labels["cur_xy"] * map_size
    tar_xy = labels["tar_xy"] * map_size

    cur_x, cur_y = cur_xy[0].item(), cur_xy[1].item()
    tar_x, tar_y = tar_xy[0].item(), tar_xy[1].item()

    # Vector predicted by NN, scaled to some reasonable length on the map
    arrow_len = map_size * 0.4
    pred_dx = cos_t * arrow_len
    pred_dy = sin_t * arrow_len

    # Convert satellite image to numpy for OpenCV (BGR, uint8)
    img_np = sat_img.cpu().permute(1, 2, 0).numpy()  # H x W x C, RGB, [0,1]
    img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

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

    # Draw NN predicted direction vector from current position (red arrow)
    end_point = (
        int(cur_x + pred_dx),
        int(cur_y + pred_dy),
    )
    cv2.arrowedLine(
        img_bgr,
        (int(cur_x), int(cur_y)),
        end_point,
        color=(0, 0, 255),
        thickness=4,
        tipLength=0.05,
    )

    # Optional: text labels
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

    cv2.imwrite(args.save_path, img_bgr)
    print(f"Saved visualization to {args.save_path}")


if __name__ == "__main__":
    main()


