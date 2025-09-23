from __future__ import annotations

import torch
from torch import nn

# Support both package and direct script execution
try:
    from .UNet_blocks import InputConv, DownConv, UpConv, OutConv
except ImportError:
    from UNet_blocks import InputConv, DownConv, UpConv, OutConv


class UNet(nn.Module):
    """1D U-Net for EEG denoising with encoder–decoder skip connections.

    - Concatenates encoder features to decoder at matching scales
    - Configurable channel widths via `widths`
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        widths: list[int] | None = None,
        down_kernels: list[int] | None = None,
        up_kernels: list[int] | None = None,
        dropout_ps: list[float] | None = None,
        use_skips: bool = True,
    ) -> None:
        super().__init__()
        
        if widths is None or len(widths) < 2:
            raise ValueError("widths must be provided and contain at least two entries")
        if down_kernels is None or up_kernels is None:
            raise ValueError("Provide down_kernels and up_kernels (same length as widths).")
        if not (len(down_kernels) == len(up_kernels) == len(widths)):
            raise ValueError("Expected len(widths) == len(down_kernels) == len(up_kernels)")
        
        # Dropout per level (0..L). Default to 0.2 if not provided.
        if dropout_ps is None:
            dropout_ps = [0.2 for _ in widths]
        if len(dropout_ps) != len(widths):
            raise ValueError("dropout_ps must have same length as widths")
        
        self.use_skips = use_skips

        # In
        self.input =  InputConv(in_channels, widths[0], kernel_size=down_kernels[0], dropout_p=dropout_ps[0])

        # Encoder
        self.encoder = nn.ModuleList(
            [
                # Map dropout to the output level (i+1)
                DownConv(widths[i], widths[i + 1], kernel_size=down_kernels[i + 1], dropout_p=dropout_ps[i + 1]) 
                for i in range(0, len(widths) - 1, 1)]
        )

        # Decoder
        self.decoder = nn.ModuleList(
            [
                UpConv(
                    in_channels=widths[j],
                    skip_channels=(widths[j - 1] if self.use_skips else 0),
                    out_channels=widths[j - 1],
                    kernel_size=up_kernels[j],
                    dropout_p=dropout_ps[j - 1],
                )
                for j in range(len(widths) - 1, 0, -1)
            ]
        )

        # Out
        self.output = OutConv(widths[0], out_channels, kernel_size=up_kernels[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path with feature collection for skips
        feats = []
        x = self.input(x)
        feats.append(x)  # level 0
        for enc in self.encoder:
            x = enc(x)
            feats.append(x)
            
        for idx, dec in enumerate(self.decoder, start=1):
            skip = feats[-(idx + 1)] if self.use_skips else None
            x = dec(x, skip)

        return self.output(x)


__all__ = ["UNet"]


if __name__ == "__main__":
    # Helper to export a default UNet to ONNX for Netron
    def export_default_onnx(out_path: str = "UNet.onnx") -> None:
        in_ch, out_ch = 30, 30
        widths = [16, 32, 64, 256, 512]
        down_k = [7, 7, 7, 7, 7]
        up_k = [7, 7, 7, 7, 7]
        drop_p = [0.2, 0.2, 0.2, 0.2, 0.2]
        model = UNet(in_ch, out_ch, widths, down_kernels=down_k, up_kernels=up_k, dropout_ps=drop_p, use_skips=True)
        x = torch.randn(16, in_ch, 30000)
        torch.onnx.export(
            model.train(),
            x,
            out_path,
            input_names=["x"],
            output_names=["y"],
            opset_version=13,
            dynamic_axes={"x": {0: "batch", 2: "time"}, "y": {0: "batch", 2: "time"}},

        )
        print(f"Exported ONNX to {out_path}")

    export_default_onnx()
