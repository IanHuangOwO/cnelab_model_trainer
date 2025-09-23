from typing import Optional

import torch
from torch import nn


class DoubleConv(nn.Module):
    """(Conv1d -> BatchNorm1d -> ReLU -> Dropout) x 2.

    Switched from Sigmoid to ReLU and added Dropout1d for regularization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        dropout_p: float = 0.2,
    ) -> None:
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
    """Input stem built from DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7, dropout_p: float = 0.2) -> None:
        super().__init__()
        self.block = DoubleConv(in_channels, out_channels, kernel_size, dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownConv(nn.Module):
    """Downscale by MaxPool1d then DoubleConv (was Down)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout_p: float = 0.2):
        super().__init__()
        self.pool = nn.MaxPool1d(2)
        self.conv = DoubleConv(in_channels, out_channels, kernel_size, dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpConv(nn.Module):
    """Upscale, concatenate skip, then DoubleConv.

    - in_channels: channels of the input to be upsampled
    - skip_channels: channels from the encoder skip connection
    - out_channels: output channels after DoubleConv
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, kernel_size: int, dropout_p: float = 0.2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels, kernel_size, dropout_p)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            # Concatenate along channel dimension
            x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


__all__ = [
    "DoubleConv", 
    "InputConv", 
    "DownConv", 
    "UpConv", 
    "OutConv"
]
