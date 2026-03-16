import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torchvision.transforms as T

from create_dataset import NavigationDataset
from cur_localization_net import CurLocalizationNet, build_localization_transforms


def build_nav_transform():
    """
    Обёртка над build_localization_transforms, подходящая для NavigationDataset,
    который ожидает transform(sat_img, cur_tile, tar_tile).
    Третий выход (tar_tile) нам не нужен, но его нужно вернуть.
    """
    base_transform = build_localization_transforms()
    to_tensor = T.ToTensor()

    def transform(sat_img, cur_tile, tar_tile):
        sat, cur = base_transform(sat_img, cur_tile)
        # TAR не используется в задаче локализации CUR, но возвращать его нужно.
        tar = to_tensor(tar_tile)
        return sat, cur, tar

    return transform


def nll_heatmap_loss(logits: torch.Tensor, target_xy: torch.Tensor) -> torch.Tensor:
    """
    logits: (B, 1, Hf, Wf) — логиты по карте (softmax ещё не применён).
    target_xy: (B, 2) — нормализованные координаты CUR в [0,1] (как cur_xy в NavigationDataset).

    Считаем отрицательное лог‑правдоподобие: -log P(x_gt, y_gt).
    """
    b, _, hf, wf = logits.shape

    # Переводим нормализованные координаты в индексы на feature‑карте.
    # target_xy[:, 0] — x, target_xy[:, 1] — y.
    x = target_xy[:, 0] * wf
    y = target_xy[:, 1] * hf

    x_idx = torch.clamp(x.long(), 0, wf - 1)
    y_idx = torch.clamp(y.long(), 0, hf - 1)

    # Переводим задачу к обычной cross_entropy по Hf*Wf классам.
    flat_logits = logits.view(b, -1)  # (B, Hf*Wf)
    linear_idx = (y_idx * wf + x_idx).long()  # (B,)

    # F.cross_entropy сам внутри применяет log_softmax + NLLLoss.
    loss = F.cross_entropy(flat_logits, linear_idx)
    return loss


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0

    for sat_img, cur_tile, _tar_tile, labels in tqdm(loader, desc="Train", leave=False):
        sat_img = sat_img.to(device)
        cur_tile = cur_tile.to(device)
        cur_xy = labels["cur_xy"].to(device)  # (B, 2), нормализованные координаты

        optimizer.zero_grad()
        logits = model(sat_img, cur_tile)  # (B, 1, Hf, Wf)
        loss = nll_heatmap_loss(logits, cur_xy)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * sat_img.size(0)

    return running_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    running_loss = 0.0

    with torch.inference_mode():
        for sat_img, cur_tile, _tar_tile, labels in tqdm(loader, desc="Val", leave=False):
            sat_img = sat_img.to(device)
            cur_tile = cur_tile.to(device)
            cur_xy = labels["cur_xy"].to(device)

            logits = model(sat_img, cur_tile)
            loss = nll_heatmap_loss(logits, cur_xy)
            running_loss += loss.item() * sat_img.size(0)

    return running_loss / len(loader.dataset)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Обучение CurLocalizationNet для предсказания heatmap положения CUR."
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
        default="dataset/satellite_maps",
        help="Директория с исходными спутниковыми картами.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Устройство для обучения.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="cur_localization_net.pt",
        help="Куда сохранить лучшие веса модели.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    annotations = Path(args.annotations)
    map_dir = Path(args.map_dir)
    device = torch.device(args.device)

    if not annotations.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations}")

    # Датасеты и dataloader‑ы
    transform = build_nav_transform()

    train_ds = NavigationDataset(
        csv_file=str(annotations),
        map_dir=str(map_dir),
        split="train",
        transform=transform,
    )
    val_ds = NavigationDataset(
        csv_file=str(annotations),
        map_dir=str(map_dir),
        split="val",
        transform=transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Модель и оптимизатор
    model = CurLocalizationNet().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)

        print(f"  train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"  Saved best model to {args.save_path}")


if __name__ == "__main__":
    main()


