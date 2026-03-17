import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from tile_localization_net import TileLocalizationNet, build_tile_localization_transforms


class LowerTarTileDataset(Dataset):
    """
    Dataset для CSV, созданного create_lower_part_dataset.py (annotations_lower_tar.csv).

    На каждой строке:
      - map_crop_x, map_crop_y: top-left 2000x2000 crop в big image
      - tar_tile_x, tar_tile_y: центр тайла (в координатах crop)

    Возвращает:
      - map_crop tensor (3, map_size, map_size)
      - tar_tile tensor (3, tile_size, tile_size)
      - tar_xy_norm: (2,) в [0,1]
    """

    def __init__(
        self,
        csv_file: str,
        big_image: str,
        split: str | None = None,
        map_size: int = 2000,
        tile_size: int = 200,
        transform=None,
    ):
        self.df = pd.read_csv(csv_file)
        if split is not None and "split" in self.df.columns:
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        self.map_size = int(map_size)
        self.tile_size = int(tile_size)
        self.half = self.tile_size // 2

        self.transform = transform

        self.big_image_path = Path(big_image)
        if not self.big_image_path.exists():
            raise FileNotFoundError(f"Big image not found: {self.big_image_path}")

        Image.MAX_IMAGE_PIXELS = None
        self.big_img = Image.open(self.big_image_path).convert("RGB")
        self.img_w, self.img_h = self.big_img.size

        if len(self.df) == 0:
            raise RuntimeError("Dataset is empty after split filtering.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        x0 = int(row["map_crop_x"])
        y0 = int(row["map_crop_y"])

        # crop map
        crop = self.big_img.crop((x0, y0, x0 + self.map_size, y0 + self.map_size))

        tx = float(row["tar_tile_x"])
        ty = float(row["tar_tile_y"])

        tile = crop.crop(
            (
                int(tx - self.half),
                int(ty - self.half),
                int(tx + self.half),
                int(ty + self.half),
            )
        )

        tar_xy_norm = torch.tensor([tx / self.map_size, ty / self.map_size], dtype=torch.float32)

        if self.transform is not None:
            crop_t, tile_t = self.transform(crop, tile)
        else:
            from torchvision.transforms import ToTensor

            to_tensor = ToTensor()
            crop_t = to_tensor(crop)
            tile_t = to_tensor(tile)

        return crop_t, tile_t, tar_xy_norm


def ce_heatmap_loss(logits: torch.Tensor, target_xy: torch.Tensor) -> torch.Tensor:
    """
    logits: (B, 1, Hf, Wf)
    target_xy: (B, 2) normalized in [0,1]
    """
    b, _, hf, wf = logits.shape

    # Soft-target Gaussian + KL works better than one-hot CE for localization.
    x = target_xy[:, 0] * (wf - 1)
    y = target_xy[:, 1] * (hf - 1)

    yy, xx = torch.meshgrid(
        torch.arange(hf, device=logits.device, dtype=torch.float32),
        torch.arange(wf, device=logits.device, dtype=torch.float32),
        indexing="ij",
    )
    xx = xx.unsqueeze(0)
    yy = yy.unsqueeze(0)
    x0 = x.view(b, 1, 1)
    y0 = y.view(b, 1, 1)

    sigma = 1.5
    dist2 = (xx - x0) ** 2 + (yy - y0) ** 2
    target = torch.exp(-0.5 * dist2 / (sigma**2))
    target = target / torch.clamp(target.sum(dim=(1, 2), keepdim=True), min=1e-12)

    log_probs = F.log_softmax(logits.view(b, -1), dim=1).view(b, hf, wf)
    return F.kl_div(log_probs, target, reduction="batchmean")


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running = 0.0

    for map_crop, tar_tile, tar_xy in tqdm(loader, desc="Train", leave=False):
        map_crop = map_crop.to(device)
        tar_tile = tar_tile.to(device)
        tar_xy = tar_xy.to(device)

        optimizer.zero_grad()
        logits = model(map_crop, tar_tile)
        loss = ce_heatmap_loss(logits, tar_xy)
        loss.backward()
        optimizer.step()

        running += loss.item() * map_crop.size(0)

    return running / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    running = 0.0

    with torch.inference_mode():
        for map_crop, tar_tile, tar_xy in tqdm(loader, desc="Val", leave=False):
            map_crop = map_crop.to(device)
            tar_tile = tar_tile.to(device)
            tar_xy = tar_xy.to(device)

            logits = model(map_crop, tar_tile)
            loss = ce_heatmap_loss(logits, tar_xy)
            running += loss.item() * map_crop.size(0)

    return running / len(loader.dataset)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Train TileLocalizationNet on the lower-part TAR-only dataset "
            "(annotations_lower_tar.csv created from big_sample5_17.tif)."
        )
    )
    p.add_argument(
        "--annotations",
        type=str,
        default="dataset/annotations_lower_tar.csv",
        help="Path to TAR-only annotations CSV.",
    )
    p.add_argument(
        "--big-image",
        type=str,
        default="big_sample5_17.tif",
        help="Path to the big image used to generate the CSV.",
    )
    p.add_argument("--map-size", type=int, default=2000)
    p.add_argument("--tile-size", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--save-path",
        type=str,
        default="tile_localization_lower_tar.pt",
        help="Where to save best model weights.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    ann = Path(args.annotations)
    if not ann.exists():
        raise FileNotFoundError(f"Annotations not found: {ann}")

    transform_pair = build_tile_localization_transforms()

    train_ds = LowerTarTileDataset(
        csv_file=str(ann),
        big_image=args.big_image,
        split="train",
        map_size=args.map_size,
        tile_size=args.tile_size,
        transform=transform_pair,
    )
    val_ds = LowerTarTileDataset(
        csv_file=str(ann),
        big_image=args.big_image,
        split="val",
        map_size=args.map_size,
        tile_size=args.tile_size,
        transform=transform_pair,
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

    model = TileLocalizationNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        tr = train_one_epoch(model, train_loader, optimizer, device)
        va = evaluate(model, val_loader, device)
        print(f"  train loss: {tr:.4f} | val loss: {va:.4f}")

        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), args.save_path)
            print(f"  Saved best model to {args.save_path}")


if __name__ == "__main__":
    main()


