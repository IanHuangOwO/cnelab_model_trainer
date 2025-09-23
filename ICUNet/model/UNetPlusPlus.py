from __future__ import annotations

from typing import List

import torch
from torch import nn

# Support both package and direct script execution
try:
    from .UNetPlusPlus_blocks import InputConv, DownConv, UpCatConvPP, OutConv
except ImportError:
    from UNetPlusPlus_blocks import InputConv, DownConv, UpCatConvPP, OutConv


class UNetPlusPlus(nn.Module):
    """1D UNet++ (Nested U-Net) in the ICUNet style.

    - Encoder built from `InputConv` + repeated `DownConv`
    - Nested decoder nodes X[i,j] with dense skip concatenations
    - Uses 1D upsampling (linear) and DoubleConv blocks with Sigmoid activations

    widths: channels per level from top (i=0) to bottom (i=L)
    down_kernels: per-level kernel sizes, len == len(widths); index 0 is input stem
    up_kernels: per-level kernel sizes, len == len(widths); index 0 is output head, indices 1..L used for up paths
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        widths: List[int] | None = None,
        down_kernels: List[int] | None = None,
        up_kernels: List[int] | None = None,
        dropout_ps: List[float] | None = None,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()

        if widths is None or len(widths) < 2:
            raise ValueError("widths must be provided and contain at least two entries")
        if down_kernels is None or up_kernels is None:
            raise ValueError("Provide down_kernels and up_kernels (same length as widths).")
        if not (len(down_kernels) == len(up_kernels) == len(widths)):
            raise ValueError("Expected len(widths) == len(down_kernels) == len(up_kernels)")

        self.widths = widths
        L = len(widths) - 1  # depth
        self.deep_supervision = deep_supervision

        if dropout_ps is None:
            dropout_ps = [0.2 for _ in widths]
        if len(dropout_ps) != len(widths):
            raise ValueError("dropout_ps must have same length as widths")

        # In
        self.input = InputConv(in_channels, widths[0], kernel_size=down_kernels[0], dropout_p=dropout_ps[0])

        # Encoder
        self.encoder = nn.ModuleList(
            [DownConv(widths[i], widths[i + 1], kernel_size=down_kernels[i + 1], dropout_p=dropout_ps[i + 1]) for i in range(L)]
        )

        # Nested decoder blocks: for each level i, create blocks for j=1..L-i
        # Access as self.decoder[i][j-1] produces X[i,j]
        dec_rows: List[nn.ModuleList] = []
        for i in range(0, L):
            row = nn.ModuleList()
            for j in range(1, L - i + 1):
                skip_channels = j * widths[i]           # concat of X[i,0..j-1]
                up_channels = widths[i + 1]             # from X[i+1, j-1]
                ksize = up_kernels[i + 1]               # per-level up kernel
                row.append(UpCatConvPP(skip_channels, up_channels, widths[i], kernel_size=ksize, dropout_p=dropout_ps[i]))
            dec_rows.append(row)
        self.decoder: List[nn.ModuleList] = nn.ModuleList(dec_rows)

        # Output heads
        self.out_head = OutConv(widths[0], out_channels, kernel_size=up_kernels[0])

        if self.deep_supervision:
            # Heads for X[0,1], X[0,2], ..., X[0,L]
            self.ds_heads = nn.ModuleList(
                [OutConv(widths[0], out_channels, kernel_size=up_kernels[0]) for _ in range(1, L + 1)]
            )

    def forward(self, x: torch.Tensor):
        widths = self.widths
        L = len(widths) - 1

        # X[i][j] storage; j ranges 0..L-i
        X: List[List[torch.Tensor]] = [[None for _ in range(L - i + 1)] for i in range(L + 1)]  # type: ignore

        # Level 0 (input stem)
        X[0][0] = self.input(x)

        # Down path to populate X[i][0]
        cur = X[0][0]
        for i, enc in enumerate(self.encoder, start=0):
            if i == 0:
                cur = enc(cur)
            else:
                cur = enc(cur)
            X[i + 1][0] = cur

        # Nested decoder to compute X[i][j] for j >= 1
        for j in range(1, L + 1):
            for i in range(0, L - j + 1):
                prev_nodes = [X[i][k] for k in range(0, j)]
                up_src = X[i + 1][j - 1]
                block = self.decoder[i][j - 1]
                X[i][j] = block(prev_nodes, up_src)

        # Output
        if self.deep_supervision:
            outs = []
            for j in range(1, L + 1):
                outs.append(self.ds_heads[j - 1](X[0][j]))
            return outs
        else:
            return self.out_head(X[0][L])


__all__ = ["UNetPlusPlus"]


if __name__ == "__main__":
    # Helper to export a default UNetPlusPlus to ONNX for Netron
    def export_default_onnx(out_path: str = "UNetPlusPlus.onnx") -> None:
        in_ch, out_ch = 30, 30
        widths = [16, 32, 64, 128, 256]
        down_k = [7, 7, 7, 7, 7]
        up_k = [7, 7, 7, 7, 7]
        drop_p = [0.2, 0.2, 0.2, 0.2, 0.2]
        model = UNetPlusPlus(in_ch, out_ch, widths, down_kernels=down_k, up_kernels=up_k, dropout_ps=drop_p, deep_supervision=False)
        model.eval()
        x = torch.randn(1, in_ch, 256)
        torch.onnx.export(
            model,
            x,
            out_path,
            input_names=["x"],
            output_names=["y"],
            opset_version=13,
            dynamic_axes={"x": {0: "batch", 2: "time"}, "y": {0: "batch", 2: "time"}},
        )
        print(f"Exported ONNX to {out_path}")

    export_default_onnx()
