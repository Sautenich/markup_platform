import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from tqdm.auto import tqdm
from PIL import Image
import pandas as pd
import numpy as np


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class TileEncoder(nn.Module):
    """
    Encodes a 200x200 tile into a vector embedding.
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


class TrajectoryNet(nn.Module):
    """
    Predicts (cos, sin) направления cur -> tar по последовательности тайлов:
      prev4, prev3, prev2, prev1, cur, а также таргетному тайлу (tar).
    """

    def __init__(self, tile_channels: int = 3, base_channels: int = 32, dropout: float = 0.3):
        super().__init__()
        self.tile_encoder = TileEncoder(tile_channels, base_channels)
        feat_dim = base_channels * 4  # 128 при base_channels=32

        self.seq_len = 5  # prev1..4 + cur
        self.gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=feat_dim,
            num_layers=1,
            batch_first=True,
        )

        # Вектор для головы: last_hidden (seq) + tar_vec
        total_dim = feat_dim * 2
        self.head = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, tile_seq: torch.Tensor, tar_tile: torch.Tensor) -> torch.Tensor:
        """
        tile_seq: (B, L=5, C, H, W)
        tar_tile: (B, C, H, W)
        """
        b, l, c, h, w = tile_seq.shape
        assert l == self.seq_len, f"Expected seq_len={self.seq_len}, got {l}"

        # Encode each tile in sequence
        seq_flat = tile_seq.view(b * l, c, h, w)
        seq_emb = self.tile_encoder(seq_flat)  # (B*L, D)
        feat_dim = seq_emb.shape[1]
        seq_emb = seq_emb.view(b, l, feat_dim)  # (B, L, D)

        # GRU over sequence, take last hidden state
        _, h_last = self.gru(seq_emb)  # h_last: (1, B, D)
        h_last = h_last.squeeze(0)  # (B, D)

        # Encode target tile
        tar_vec = self.tile_encoder(tar_tile)  # (B, D)

        x = torch.cat([h_last, tar_vec], dim=1)  # (B, 2D)
        return self.head(x)


class TrajectoryDataset(Dataset):
    """
    Dataset для trajectory-задачи из annotations_manual_interest_traj.csv.

    Каждый элемент:
      - последовательность из 4 предыдущих тайлов + cur (в координатах кропа 2000x2000)
      - таргетный тайл
      - вектор dx, dy (cur -> tar) из CSV
    """

    def __init__(
        self,
        csv_file: str,
        map_path: str,
        map_size: int = 2000,
        tile_size: int = 200,
        split: str | None = None,
    ):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        if split is not None and "split" in self.df.columns:
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        self.map_size = int(map_size)
        self.tile_size = int(tile_size)
        self.half = self.tile_size // 2

        self.map_path = Path(map_path)
        if not self.map_path.exists():
            raise FileNotFoundError(f"Map image not found: {self.map_path}")

        Image.MAX_IMAGE_PIXELS = None
        self.big_img = Image.open(self.map_path).convert("RGB")

        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.prev_cols = [
            ("prev1_tile_x", "prev1_tile_y"),
            ("prev2_tile_x", "prev2_tile_y"),
            ("prev3_tile_x", "prev3_tile_y"),
            ("prev4_tile_x", "prev4_tile_y"),
        ]

    def __len__(self) -> int:
        return len(self.df)

    def _crop_tile(self, crop_img: Image.Image, cx: float, cy: float) -> torch.Tensor:
        # crop_img: PIL map_size x map_size
        x0 = int(cx - self.half)
        y0 = int(cy - self.half)
        x1 = x0 + self.tile_size
        y1 = y0 + self.tile_size
        tile = crop_img.crop((x0, y0, x1, y1))
        t = self.to_tensor(tile)
        t = self.normalize(t)
        return t

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        x0 = int(row["map_crop_x"])
        y0 = int(row["map_crop_y"])

        # Кроп карты 2000x2000
        crop = self.big_img.crop((x0, y0, x0 + self.map_size, y0 + self.map_size))

        # Собираем последовательность prev4..prev1..cur (если prev отсутствуют, используем cur)
        seq_centers = []
        for cx_col, cy_col in self.prev_cols:
            if cx_col in row and cy_col in row and not pd.isna(row[cx_col]) and not pd.isna(row[cy_col]):
                px = float(row[cx_col])
                py = float(row[cy_col])
            else:
                px = float(row["cur_tile_x"])
                py = float(row["cur_tile_y"])
            seq_centers.append((px, py))

        cur_x = float(row["cur_tile_x"])
        cur_y = float(row["cur_tile_y"])
        tar_x = float(row["tar_tile_x"])
        tar_y = float(row["tar_tile_y"])

        seq_centers.append((cur_x, cur_y))  # последний — текущий

        tiles = [self._crop_tile(crop, cx, cy) for (cx, cy) in seq_centers]
        tile_seq = torch.stack(tiles, dim=0)  # (L=5, C, H, W)

        tar_tile = self._crop_tile(crop, tar_x, tar_y)

        dx = float(row["vector_dx"])
        dy = float(row["vector_dy"])
        vec = torch.tensor([dx, dy], dtype=torch.float32)

        return tile_seq, tar_tile, vec


def vector_to_angle_components(vector: torch.Tensor) -> torch.Tensor:
    dx = vector[:, 0]
    dy = vector[:, 1]
    theta = torch.atan2(dy, dx)
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.MSELoss()
    running_loss = 0.0

    for tile_seq, tar_tile, vec in tqdm(loader, desc="Train", leave=False):
        tile_seq = tile_seq.to(device)  # (B, L, C, H, W)
        tar_tile = tar_tile.to(device)
        vec = vec.to(device)

        target = vector_to_angle_components(vec)

        optimizer.zero_grad()
        pred = model(tile_seq, tar_tile)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * tile_seq.size(0)

    return running_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    criterion = nn.MSELoss()
    running_loss = 0.0

    with torch.inference_mode():
        for tile_seq, tar_tile, vec in tqdm(loader, desc="Val", leave=False):
            tile_seq = tile_seq.to(device)
            tar_tile = tar_tile.to(device)
            vec = vec.to(device)

            target = vector_to_angle_components(vec)
            pred = model(tile_seq, tar_tile)
            loss = criterion(pred, target)
            running_loss += loss.item() * tile_seq.size(0)

    return running_loss / len(loader.dataset)


def parse_train_args():
    parser = argparse.ArgumentParser(description="Train trajectory azimuth network on manual_interest_traj dataset.")
    parser.add_argument(
        "--annotations",
        type=str,
        default="dataset/annotations_manual_interest_traj.csv",
        help="Path to trajectory annotations CSV.",
    )
    parser.add_argument(
        "--map-image",
        type=str,
        default="big_sample5_17.tif",
        help="Path to the large source image used to generate the annotations.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--save-path",
        type=str,
        default="azimuth_net_trajectory.pt",
        help="Where to save best model weights.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of dataset to use for validation (simple random split).",
    )
    return parser.parse_args()


def main():
    args = parse_train_args()

    ann_path = Path(args.annotations)
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {ann_path}")

    # Простое случайное разделение на train/val по индексам
    full_df = pd.read_csv(ann_path)
    n = len(full_df)
    if n == 0:
        raise RuntimeError("Empty annotations file.")

    indices = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(indices)

    val_size = int(max(1, n * args.val_fraction))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_df = full_df.iloc[train_idx].reset_index(drop=True)
    val_df = full_df.iloc[val_idx].reset_index(drop=True)

    # Временные CSV для train/val
    tmp_train = ann_path.parent / "_traj_train_tmp.csv"
    tmp_val = ann_path.parent / "_traj_val_tmp.csv"
    train_df.to_csv(tmp_train, index=False)
    val_df.to_csv(tmp_val, index=False)

    device = torch.device(args.device)

    train_ds = TrajectoryDataset(
        csv_file=str(tmp_train),
        map_path=args.map_image,
        map_size=2000,
        tile_size=200,
    )
    val_ds = TrajectoryDataset(
        csv_file=str(tmp_val),
        map_path=args.map_image,
        map_size=2000,
        tile_size=200,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    model = TrajectoryNet(dropout=args.dropout).to(device)
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


