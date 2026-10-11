"""Compact U-Net model for binary crack segmentation."""

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(3, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.enc4 = ConvBlock(c * 4, c * 8)
        self.bridge = ConvBlock(c * 8, c * 16)
        self.pool = nn.MaxPool2d(2)
        self.dec4 = ConvBlock(c * 16 + c * 8, c * 8)
        self.dec3 = ConvBlock(c * 8 + c * 4, c * 4)
        self.dec2 = ConvBlock(c * 4 + c * 2, c * 2)
        self.dec1 = ConvBlock(c * 2 + c, c)
        self.head = nn.Conv2d(c, 1, kernel_size=1)

    @staticmethod
    def _up(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat((skip, x), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pad_h, pad_w = (-h) % 16, (-w) % 16
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bridge(self.pool(e4))
        d4 = self.dec4(self._up(b, e4))
        d3 = self.dec3(self._up(d4, e3))
        d2 = self.dec2(self._up(d3, e2))
        d1 = self.dec1(self._up(d2, e1))
        return self.head(d1)[..., :h, :w]


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (prob * target).sum(dims)
    dice = (2 * intersection + 1.0) / (prob.sum(dims) + target.sum(dims) + 1.0)
    return bce + (1.0 - dice).mean()
