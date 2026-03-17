import sys
from pathlib import Path as _Path

# Allow running as a script: `python visualization/visualize_cur_localization_heatmap.py`
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
from pathlib import Path

import cv2
import numpy as np

import torch
import torchvision.transforms as T

from data_generation.create_dataset import NavigationDataset
from training_and_NN.cur_localization_net import CurLocalizationNet, build_localization_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Визуализация heatmap‑распределения положения CUR по карте "
            "для сети CurLocalizationNet."
        )
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="dataset/annotations.csv",
        help="Путь к annotations.csv.",
    )
    parser.add_argument(
        "--map-dir",
        type=str,
        default="dataset/satellite_maps/1000x1000_square",
        help="Директория с исходными спутниковыми картами.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="weights/cur_localization_net.pt",
        help="Путь к обученным весам CurLocalizationNet (опционально).",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=-1,
        help="Индекс примера из test‑сплита для визуализации. Если -1, выбрать случайный.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Устройство для инференса.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="viz_cur_heatmap.png",
        help="Куда сохранить картинку с heatmap.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    annotations = Path(args.annotations)
    map_dir = Path(args.map_dir)
    device = torch.device(args.device)

    # Берём dataset без нормализации, чтобы удобнее визуализировать.
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

    sat_img, cur_tile, _tar_tile, labels = dataset[idx]

    # sat_img, cur_tile — тензоры [0,1]; для модели нужно нормировать так же, как при обучении.
    transform_pair = build_localization_transforms()

    to_pil = T.ToPILImage()
    sat_pil = to_pil(sat_img)
    cur_pil = to_pil(cur_tile)

    sat_in, cur_in = transform_pair(sat_pil, cur_pil)
    sat_in = sat_in.unsqueeze(0).to(device)
    cur_in = cur_in.unsqueeze(0).to(device)

    # Модель
    model = CurLocalizationNet().to(device)
    model_path = Path(args.model_path)
    if model_path.exists():
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded model weights from {model_path}")
    else:
        print(f"Model weights not found at {model_path}, using randomly initialized model.")

    model.eval()
    with torch.inference_mode():
        logits = model(sat_in, cur_in)[0, 0]  # (Hf, Wf)

    heatmap = torch.softmax(logits.view(-1), dim=0).view_as(logits)
    heatmap_np = heatmap.cpu().numpy()

    h_map, w_map = sat_img.shape[1], sat_img.shape[2]

    lo, hi = np.percentile(heatmap_np, [1.0, 99.5])
    heatmap_np = np.clip((heatmap_np - lo) / max(hi - lo, 1e-12), 0.0, 1.0)

    heatmap_resized = cv2.resize(
        heatmap_np,
        (w_map, h_map),
        interpolation=cv2.INTER_CUBIC,
    )

    img_np = sat_img.cpu().permute(1, 2, 0).numpy()
    img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    heatmap_color = cv2.applyColorMap(
        (heatmap_resized * 255.0).astype(np.uint8),
        cv2.COLORMAP_JET,
    )

    overlay = cv2.addWeighted(img_bgr, 0.5, heatmap_color, 0.5, 0)

    map_size = dataset.map_size
    tile_size = dataset.tile_size
    cur_xy = labels["cur_xy"] * map_size
    cur_x, cur_y = cur_xy[0].item(), cur_xy[1].item()
    half = tile_size // 2

    cur_tl = (int(cur_x - half), int(cur_y - half))
    cur_br = (int(cur_x + half), int(cur_y + half))
    cv2.rectangle(overlay, cur_tl, cur_br, (0, 255, 0), 2)
    cv2.putText(
        overlay,
        "CUR",
        (cur_tl[0] + 5, cur_tl[1] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

    cv2.imwrite(args.save_path, overlay)
    print(f"Saved CUR localization heatmap to {args.save_path}")


if __name__ == "__main__":
    main()

