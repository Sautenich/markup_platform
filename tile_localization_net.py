import torch
import torch.nn as nn
import torch.nn.functional as F


class TileEncoder(nn.Module):
    """
    Кодирует тайл 200x200 в вектор признаков.
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
    Кодирует crop карты (например, 2000x2000) в пространственную карту признаков (B, C, H, W),
    без глобального pooling — чтобы по HxW можно было строить heatmap.
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


class TileLocalizationNet(nn.Module):
    """
    Универсальная локализация тайла на карте:

    Вход:
      - map_crop: (B, 3, H, W)
      - query_tile: (B, 3, h, w)

    Выход:
      - logits: (B, 1, Hf, Wf) — логиты по spatial‑позициям (softmax применять в лоссе/визуализации)

    Идея та же, что в cur_localization_net.py:
      map_encoder(map) -> (B, C, Hf, Wf)
      tile_encoder(tile) -> (B, C)
      logits_ij = <tile_vec, map_feat_ij>
    """

    def __init__(
        self,
        map_channels: int = 3,
        tile_channels: int = 3,
        base_channels: int = 32,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.map_encoder = MapEncoder(map_channels, base_channels)
        self.tile_encoder = TileEncoder(tile_channels, base_channels)
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    def forward(self, map_crop: torch.Tensor, query_tile: torch.Tensor) -> torch.Tensor:
        map_feat = self.map_encoder(map_crop)  # (B, C, Hf, Wf)
        b, c, hf, wf = map_feat.shape
        n = hf * wf

        tile_vec = self.tile_encoder(query_tile)  # (B, C)
        map_flat = map_feat.view(b, c, n)  # (B, C, N)
        map_flat = F.normalize(map_flat, dim=1, eps=1e-6)
        tile_vec = F.normalize(tile_vec, dim=1, eps=1e-6)
        scores = torch.bmm(tile_vec.unsqueeze(1), map_flat).squeeze(1)  # (B, N)
        scores = scores / torch.clamp(self.temperature, min=1e-6)
        logits = scores.view(b, 1, hf, wf)
        return logits


def build_tile_localization_transforms():
    """
    Нормализация такая же, как в остальных моделях.
    """
    from torchvision import transforms as T

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = T.ToTensor()

    def transform_pair(map_crop, query_tile):
        m = normalize(to_tensor(map_crop))
        t = normalize(to_tensor(query_tile))
        return m, t

    return transform_pair


