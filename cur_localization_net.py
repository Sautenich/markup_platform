import torch
import torch.nn as nn
import torch.nn.functional as F


class TileEncoder(nn.Module):
    """
    Простая CNN‑сетка, кодирующая тайл 200x200 в вектор признаков.
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
    Кодирует crop карты (например, 1000x1000) в пространственную карту признаков (B, C, H, W),
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


class CurLocalizationNet(nn.Module):
    """
    Модель, которая по (map_crop, cur_tile) выдаёт распределение вероятностей по карте,
    где может находиться CUR (heatmap над HxW).

    Вариант реализации:
      1) MapEncoder -> карта признаков (B, C, H, W)
      2) TileEncoder(cur) -> вектор (B, C)
      3) Косинусное сходство query (cur_vec) с каждым spatial‑вектором карты
      4) Softmax по всем H*W -> нормализованная heatmap.
    """

    def __init__(
        self,
        map_channels: int = 3,
        tile_channels: int = 3,
        base_channels: int = 32,
    ):
        super().__init__()
        self.map_encoder = MapEncoder(map_channels, base_channels)
        self.tile_encoder = TileEncoder(tile_channels, base_channels)

    def forward(self, sat_img: torch.Tensor, cur_tile: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sat_img: (B, 3, H, W) — crop карты
            cur_tile: (B, 3, h, w) — вырезанный тайл CUR

        Returns:
            logits: (B, 1, H_feat, W_feat) — "сырые" логиты по карте, без softmax.
        """
        map_feat = self.map_encoder(sat_img)  # (B, C, Hf, Wf)
        b, c, hf, wf = map_feat.shape
        n = hf * wf

        cur_vec = self.tile_encoder(cur_tile)  # (B, C)

        # Преобразуем карту в (B, C, N) и считаем скалярное произведение с вектором CUR.
        # Без нормализации и дополнительного масштаба — логиты могут иметь
        # достаточно большую амплитуду, чтобы softmax не был почти равномерным.
        map_flat = map_feat.view(b, c, n)  # (B, C, N)
        scores = torch.bmm(cur_vec.unsqueeze(1), map_flat).squeeze(1)  # (B, N)

        # Возвращаем логиты, softmax будет применён в лоссе / при визуализации.
        logits = scores.view(b, 1, hf, wf)
        return logits


def build_localization_transforms():
    """
    Нормализация такая же, как в других сетках (ImageNet‑подобная).
    """
    from torchvision import transforms as T

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = T.ToTensor()

    def transform_pair(sat_img, cur_tile):
        sat = normalize(to_tensor(sat_img))
        cur = normalize(to_tensor(cur_tile))
        return sat, cur

    return transform_pair


