import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm.auto import tqdm

from create_dataset import NavigationDataset


class TileEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 60 -> 30
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 30 -> 15
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x.view(x.size(0), -1)


class MapEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 1000 -> 250
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 250 -> 125
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(5, 5),  # 125 -> 25
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x.view(x.size(0), -1)


class AzimuthNet(nn.Module):
    """
    Модель, которая получает:
      - sat_img (спутниковый кроп 1000x1000),
      - cur_tile (текущий тайл),
      - tar_tile (тайл цели),
    и предсказывает азимут (направление cur -> tar).

    Чтобы обойти проблему периодичности угла, сеть предсказывает (cos(theta), sin(theta)).
    """

    def __init__(self, map_channels: int = 3, tile_channels: int = 3, base_channels: int = 32):
        super().__init__()
        self.map_encoder = MapEncoder(map_channels, base_channels)
        self.tile_encoder = TileEncoder(tile_channels, base_channels)

        map_feat_dim = base_channels * 4
        tile_feat_dim = base_channels * 4
        total_dim = map_feat_dim + tile_feat_dim * 2

        self.head = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),  # cos, sin
        )

    def forward(self, sat_img: torch.Tensor, cur_tile: torch.Tensor, tar_tile: torch.Tensor) -> torch.Tensor:
        map_feat = self.map_encoder(sat_img)
        cur_feat = self.tile_encoder(cur_tile)
        tar_feat = self.tile_encoder(tar_tile)

        x = torch.cat([map_feat, cur_feat, tar_feat], dim=1)
        out = self.head(x)
        return out


def build_transforms():
    """
    Build transforms for training/validation.
    Uses standard ImageNet normalization.
    """
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def transform_triplet(sat_img, cur_tile, tar_tile):
        to_tensor = T.ToTensor()
        sat = normalize(to_tensor(sat_img))
        cur = normalize(to_tensor(cur_tile))
        tar = normalize(to_tensor(tar_tile))
        return sat, cur, tar

    return transform_triplet


def make_dataloaders(
    annotations: Path,
    map_dir: Path,
    batch_size: int,
    num_workers: int = 4,
):
    transform = build_transforms()

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
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader


def vector_to_angle_components(vector: torch.Tensor) -> torch.Tensor:
    """
    vector: (B, 2) где [dx, dy]
    Возвращает (B, 2): [cos(theta), sin(theta)].
    """
    dx = vector[:, 0]
    dy = vector[:, 1]
    theta = torch.atan2(dy, dx)  # в радианах
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    return torch.stack([cos_t, sin_t], dim=1)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.MSELoss()
    running_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        sat_img, cur_tile, tar_tile, labels = batch
        sat_img = sat_img.to(device)
        cur_tile = cur_tile.to(device)
        tar_tile = tar_tile.to(device)

        vec = labels["vector"].to(device)  # (B, 2): dx, dy
        target = vector_to_angle_components(vec)

        optimizer.zero_grad()
        pred = model(sat_img, cur_tile, tar_tile)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * sat_img.size(0)

    return running_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    criterion = nn.MSELoss()
    running_loss = 0.0

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Val", leave=False):
            sat_img, cur_tile, tar_tile, labels = batch
            sat_img = sat_img.to(device)
            cur_tile = cur_tile.to(device)
            tar_tile = tar_tile.to(device)

            vec = labels["vector"].to(device)
            target = vector_to_angle_components(vec)

            pred = model(sat_img, cur_tile, tar_tile)
            loss = criterion(pred, target)

            running_loss += loss.item() * sat_img.size(0)

    return running_loss / len(loader.dataset)


def parse_args():
    parser = argparse.ArgumentParser(description="Train azimuth prediction network.")
    parser.add_argument(
        "--annotations",
        type=str,
        default="dataset/annotations.csv",
        help="Путь к CSV с аннотациями.",
    )
    parser.add_argument(
        "--map-dir",
        type=str,
        default="dataset/satellite_maps",
        help="Директория с исходными спутниковыми снимками.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--save-path",
        type=str,
        default="azimuth_net.pt",
        help="Куда сохранить обученную модель.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    annotations = Path(args.annotations)
    map_dir = Path(args.map-dir) if hasattr(args, "map-dir") else Path(args.map_dir)

    device = torch.device(args.device)

    train_loader, val_loader = make_dataloaders(
        annotations=annotations,
        map_dir=map_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = AzimuthNet()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)

        print(
            f"  train loss: {train_loss:.4f} | val loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"  Saved best model to {args.save_path}")


if __name__ == "__main__":
    main()

