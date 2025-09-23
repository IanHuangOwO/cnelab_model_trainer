from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn

# Support both package and direct script execution
try:
    from .SwinUnet_blocks import InputProj, SwinStageDown, SwinStageUp, OutConv
except ImportError:
    from SwinUnet_blocks import InputProj, SwinStageDown, SwinStageUp, OutConv


class SwinUnet(nn.Module):
    """Swin U-Net in 1D, following the ICUNet style and API.

    - widths: channel sizes per scale (top → bottom)
    - depths: number of Swin blocks per scale (same length as widths; default 2 each)
    - window_size: attention window size along time axis
    - num_heads: per-scale attention heads (optional; defaults derived from widths)
    - use_skips: toggle encoder–decoder concatenation
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        widths: List[int],
        depths: Optional[List[int]] = None,
        dropout_ps: Optional[List[float]] = None,
        window_size: int = 8,
        num_heads: Optional[List[int]] = None,
        use_skips: bool = True,
    ) -> None:
        super().__init__()
        if widths is None or len(widths) < 2:
            raise ValueError("widths must be provided and contain at least two entries")

        L = len(widths) - 1
        if depths is None:
            depths = [2 for _ in widths]
        if len(depths) != len(widths):
            raise ValueError("depths must match len(widths)")

        # Derive attention heads if not given (ensure divisor of channels)
        if num_heads is None:
            num_heads = []
            for c in widths:
                h = 1
                for cand in (8, 4, 2, 1):
                    if c % cand == 0:
                        h = cand
                        break
                num_heads.append(h)
        if len(num_heads) != len(widths):
            raise ValueError("num_heads must match len(widths)")

        self.widths = widths
        self.depths = depths
        self.window_size = window_size
        self.num_heads = num_heads
        self.use_skips = use_skips
        if dropout_ps is None:
            dropout_ps = [0.0 for _ in widths]
        if len(dropout_ps) != len(widths):
            raise ValueError("dropout_ps must match len(widths)")

        # Stem
        self.input = InputProj(in_channels, widths[0])

        # Encoder stages 0..L-1 (each outputs widths[i+1])
        self.encoder = nn.ModuleList(
            [
                SwinStageDown(
                    in_channels=widths[i],
                    out_channels=(widths[i + 1] if i < L else None),
                    depth=depths[i],
                    num_heads=num_heads[i],
                    window_size=window_size,
                    dropout_p=dropout_ps[i],
                )
                for i in range(0, L)
            ]
        )

        # Bottleneck (no downsample) at level L
        self.bottleneck = SwinStageDown(
            in_channels=widths[L], out_channels=None, depth=depths[L], num_heads=num_heads[L], window_size=window_size, dropout_p=dropout_ps[L]
        )

        # Decoder stages L..1 (mirror), output widths[j-1]
        self.decoder = nn.ModuleList(
            [
                SwinStageUp(
                    in_channels=widths[j],
                    skip_channels=(widths[j - 1] if use_skips else 0),
                    out_channels=widths[j - 1],
                    depth=depths[j - 1],
                    num_heads=num_heads[j - 1],
                    window_size=window_size,
                    dropout_p=dropout_ps[j - 1],
                )
                for j in range(L, 0, -1)
            ]
        )

        # Head
        self.output = OutConv(widths[0], out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats: List[torch.Tensor] = []
        x = self.input(x)
        feats.append(x)  # level 0
        for enc in self.encoder:
            x = enc(x)
            feats.append(x)

        # bottleneck
        x = self.bottleneck(x)

        # decode with optional skips
        for idx, dec in enumerate(self.decoder, start=1):
            skip = feats[-(idx + 1)] if self.use_skips else None
            x = dec(x, skip)

        return self.output(x)


__all__ = ["SwinUnet"]


if __name__ == "__main__":
    # Helper to export a default SwinUnet to ONNX for Netron
    def export_default_onnx(out_path: str = "SwinUnet.onnx") -> None:
        in_ch, out_ch = 30, 30
        widths = [16, 32, 64]
        depths = [2, 2, 2]
        drop_p = [0.2, 0.2, 0.2]
        model = SwinUnet(in_ch, out_ch, widths=widths, depths=depths, dropout_ps=drop_p, window_size=8, use_skips=True)
        model.eval()
        x = torch.randn(1, in_ch, 256)
        torch.onnx.export(
            model,
            x,
            out_path,
            input_names=["x"],
            output_names=["y"],
            opset_version=14,
            dynamic_axes={"x": {0: "batch", 2: "time"}, "y": {0: "batch", 2: "time"}},
        )
        print(f"Exported ONNX to {out_path}")

    export_default_onnx()
