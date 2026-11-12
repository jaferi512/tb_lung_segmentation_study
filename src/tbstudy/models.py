"""Networks: four-level U-Net (1,942,289 parameters, as in the paper) and ResNet-18 with one output."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    """U-Net after Ronneberger et al. (2015): 16-32-64-128 encoder, 256-channel bottleneck [paper, Table 4]."""

    def __init__(self, in_channels: int = 1, base: int = 16):
        super().__init__()
        ch = [base * 2 ** i for i in range(5)]           # 16, 32, 64, 128, 256
        self.enc = nn.ModuleList([DoubleConv(in_channels, ch[0])] +
                                 [DoubleConv(ch[i], ch[i + 1]) for i in range(3)])
        self.bottleneck = DoubleConv(ch[3], ch[4])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(4))])
        self.dec = nn.ModuleList([DoubleConv(ch[i] * 2, ch[i]) for i in reversed(range(4))])
        self.head = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        skips = []
        for block in self.enc:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.head(x)  # logits


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """[paper, Eqs. 1-2] L = BCE + soft Dice (computed per image, averaged over the batch)."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits).flatten(1)
    g = target.flatten(1)
    dice = 1 - (2 * (p * g).sum(1) + smooth) / (p.sum(1) + g.sum(1) + smooth)
    return bce + dice.mean()


def resnet18_binary(pretrained: bool = True) -> nn.Module:
    """ImageNet-initialised ResNet-18 with the final layer replaced by a single logit [paper]."""
    from torchvision.models import ResNet18_Weights, resnet18
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
