from __future__ import annotations

import torch
from torch import nn

from .ICUNet_blocks import InputConv, DownConv, UpConv, OutConv


class ICUNet(nn.Module):
    """Simple, readable 1D U-Net for EEG denoising.

    - No skip concatenation (matches earlier project behavior)
    - Configurable channel widths via `widths`
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        widths: list[int] | None = None,
        down_kernels: list[int] | None = None,
        up_kernels: list[int] | None = None,
    ) -> None:
        super().__init__()
        if widths is None or len(widths) < 2:
            raise ValueError("widths must be provided and contain at least two entries")
        
        if down_kernels is None or up_kernels is None:
            raise ValueError(
                "Provide down_kernels and up_kernels (same length as widths). The first down kernel is in; the first up kernel is out."
            )
        if not (len(down_kernels) == len(up_kernels) == len(widths)):
            raise ValueError(
                "Expected len(widths) == len(down_kernels) == len(up_kernels). First down is in; first up is out."
            )

        # In
        setattr(self, "in", InputConv(in_channels, widths[0], kernel_size=down_kernels[0]))

        # Encoder
        self.encoder = nn.ModuleList(
            [
                DownConv(widths[i], widths[i + 1], kernel_size=down_kernels[i + 1]) 
                for i in range(0, len(widths) - 1, 1)]
        )

        # Decoder
        self.decoder = nn.ModuleList(
            [
                UpConv(widths[j], widths[j - 1], kernel_size=up_kernels[j])
                for j in range(len(widths) - 1, 0, -1)
            ]
        )

        # Out
        self.out = OutConv(widths[0], out_channels, kernel_size=up_kernels[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = getattr(self, "in")(x)
        for enc in self.encoder:
            x = enc(x)
        for dec in self.decoder:
            x = dec(x)
        return self.out(x)


__all__ = ["ICUNet"]