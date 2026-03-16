import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm.auto import tqdm

from create_dataset import NavigationDataset


class TileEncoder(nn.Module):
    """
    Encodes a 200x200 (or similar) tile into a single vector embedding.
    Pooling to 1x1 here is OK: tile -> vector.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x.view(x.size(0), -1)


class MapEncoder(nn.Module):
    """
    Encodes the 1000x1000 map crop into a spatial feature map (B, C, H, W).
    IMPORTANT: no AdaptiveAvgPool2d(1) here to preserve spatial information.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(5, 5),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def attention_pool(map_feat: torch.Tensor, query_vec: torch.Tensor) -> torch.Tensor:
    """
    Attention pooling over spatial map features using a tile vector as a query.

    Args:
        map_feat: (B, C, H, W)
        query_vec: (B, C)

    Returns:
        pooled: (B, C)
    """
    b, c, h, w = map_feat.shape
    n = h * w
    map_flat = map_feat.view(b, c, n)  # (B, C, N)

    # Stabilize attention: use scaled cosine attention (normalize query and keys).
    # This prevents very large dot-products from saturating softmax early in training.
    q = F.normalize(query_vec, dim=1)  # (B, C)
    k = F.normalize(map_flat, dim=1)  # (B, C, N) normalize per spatial position

    scale = 1.0 / (c ** 0.5)
    scores = torch.bmm(q.unsqueeze(1), k).squeeze(1) * scale  # (B, N)
    att = F.softmax(scores, dim=1)  # (B, N)

    # Weighted sum: (B, C, N) @ (B, N, 1) -> (B, C)
    pooled = torch.bmm(map_flat, att.unsqueeze(2)).squeeze(2)
    return pooled


class AzimuthNetAttention(nn.Module):
    """
    Predicts (cos(theta), sin(theta)) for the direction cur -> tar.

    Key difference vs AzimuthNet:
      - MapEncoder returns a spatial feature map
      - Tile vectors act as queries to pool map features via attention
    """

    def __init__(
        self,
        map_channels: int = 3,
        tile_channels: int = 3,
        base_channels: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.map_encoder = MapEncoder(map_channels, base_channels)
        self.tile_encoder = TileEncoder(tile_channels, base_channels)

        feat_dim = base_channels * 4

        # We use 4 vectors: pooled_cur, pooled_tar, cur_vec, tar_vec
        total_dim = feat_dim * 4

        self.head = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(
        self, sat_img: torch.Tensor, cur_tile: torch.Tensor, tar_tile: torch.Tensor
    ) -> torch.Tensor:
        map_feat = self.map_encoder(sat_img)  # (B, C, H, W)
        cur_vec = self.tile_encoder(cur_tile)  # (B, C)
        tar_vec = self.tile_encoder(tar_tile)  # (B, C)

        pooled_cur = attention_pool(map_feat, cur_vec)  # (B, C)
        pooled_tar = attention_pool(map_feat, tar_vec)  # (B, C)

        x = torch.cat([pooled_cur, pooled_tar, cur_vec, tar_vec], dim=1)
        return self.head(x)


def build_transforms(augment_tiles: bool = False):
    """
    Normalization matches the original training.
    Optional: light photometric augmentations on tiles only.
    """
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = T.ToTensor()

    # These are photometric only (no geometry), so we can apply them independently.
    tile_aug = T.Compose(
        [
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.RandomGrayscale(p=0.1),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ]
    )

    def transform_triplet(sat_img, cur_tile, tar_tile):
        sat = normalize(to_tensor(sat_img))
        if augment_tiles:
            cur_tile = tile_aug(cur_tile)
            tar_tile = tile_aug(tar_tile)
        cur = normalize(to_tensor(cur_tile))
        tar = normalize(to_tensor(tar_tile))
        return sat, cur, tar

    return transform_triplet


def make_dataloaders(
    annotations: Path,
    map_dir: Path,
    batch_size: int,
    num_workers: int = 4,
    augment_tiles: bool = False,
):
    transform = build_transforms(augment_tiles=augment_tiles)

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
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def vector_to_angle_components(vector: torch.Tensor) -> torch.Tensor:
    dx = vector[:, 0]
    dy = vector[:, 1]
    theta = torch.atan2(dy, dx)
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.MSELoss()
    running_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        sat_img, cur_tile, tar_tile, labels = batch
        sat_img = sat_img.to(device)
        cur_tile = cur_tile.to(device)
        tar_tile = tar_tile.to(device)

        vec = labels["vector"].to(device)
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
    parser = argparse.ArgumentParser(description="Train azimuth prediction network with attention pooling.")
    parser.add_argument(
        "--annotations",
        type=str,
        default="dataset/annotations.csv",
        help="Path to annotations CSV.",
    )
    parser.add_argument(
        "--map-dir",
        type=str,
        default="dataset/satellite_maps",
        help="Directory with original satellite images.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--augment-tiles", action="store_true", help="Enable photometric augmentations on tiles.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--save-path",
        type=str,
        default="azimuth_net_attention.pt",
        help="Where to save best model weights.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    annotations = Path(args.annotations)
    map_dir = Path(args.map_dir)
    device = torch.device(args.device)

    train_loader, val_loader = make_dataloaders(
        annotations=annotations,
        map_dir=map_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment_tiles=args.augment_tiles,
    )

    model = AzimuthNetAttention(dropout=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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


