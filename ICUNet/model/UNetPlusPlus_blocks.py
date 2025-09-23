from typing import List

import torch
from torch import nn


class DoubleConv(nn.Module):
    """(Conv1d -> BatchNorm1d -> ReLU -> Dropout1d) x 2."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7, dropout_p: float = 0.2) -> None:
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout1d(p=dropout_p),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout1d(p=dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InputConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7, dropout_p: float = 0.2) -> None:
        super().__init__()
        self.block = DoubleConv(in_channels, out_channels, kernel_size, dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownConv(nn.Module):
    """Downscale by MaxPool1d then DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout_p: float = 0.2) -> None:
        super().__init__()
        self.pool = nn.MaxPool1d(2)
        self.conv = DoubleConv(in_channels, out_channels, kernel_size, dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpCatConvPP(nn.Module):
    """UNet++ upsampling block: upsample deeper feature, concatenate with previous nodes, then DoubleConv.

    - skip_channels: total channels from concatenated previous nodes at this level (sum of X[i,0..j-1])
    - up_channels: channels of the upsampled deeper node X[i+1, j-1]
    - out_channels: output channels (widths[i])
    """

    def __init__(self, skip_channels: int, up_channels: int, out_channels: int, kernel_size: int, dropout_p: float = 0.2) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        in_channels = skip_channels + up_channels
        self.conv = DoubleConv(in_channels, out_channels, kernel_size, dropout_p)

    def forward(self, prev_nodes: List[torch.Tensor], up_from_below: torch.Tensor) -> torch.Tensor:
        y = self.up(up_from_below)
        if prev_nodes:
            y = torch.cat([*prev_nodes, y], dim=1)
        return self.conv(y)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


__all__ = [
    "DoubleConv",
    "InputConv",
    "DownConv",
    "UpCatConvPP",
    "OutConv",
]
