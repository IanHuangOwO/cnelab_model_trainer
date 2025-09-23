from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn

# Support both package and direct script execution
try:
    from .UNETR_blocks import (
        PatchEmbed1D,
        ViTEncoder1D,
        UpConvUNETR,
        OutConv,
    )
except ImportError:
    from UNETR_blocks import (
        PatchEmbed1D,
        ViTEncoder1D,
        UpConvUNETR,
        OutConv,
    )


class UNETR(nn.Module):
    """UNETR-style 1D model in IC-UNet style.

    - Transformer encoder over patch embeddings (ViT-like)
    - Decoder of upsampling conv blocks with optional skip connections
    - Skips come from intermediate transformer layers, projected and upsampled to match

    Args
    - widths: channels per scale (top→bottom), len >= 2. widths[-1] is bottleneck channels
    - patch_size: stride and kernel for patch embedding
    - embed_dim: transformer embedding dimension
    - depth: number of transformer blocks
    - num_heads: multi-head attention heads
    - mlp_ratio: MLP expansion in transformer
    - use_skips: toggle using transformer intermediate states as skips
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        widths: List[int],
        dropout_ps: List[float],
        patch_size: int = 8,
        embed_dim: int = 128,
        depth: int = 8,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        use_skips: bool = True,
    ) -> None:
        super().__init__()
        if widths is None or len(widths) < 2:
            raise ValueError("widths must be provided and contain at least two entries")

        self.widths = widths
        self.use_skips = use_skips
        L = len(widths) - 1

        if dropout_ps is None:
            dropout_ps = [0.2 for _ in widths]
        if len(dropout_ps) != len(widths):
            raise ValueError("dropout_ps must have same length as widths")

        # Patch embed + linear projection to embed_dim
        self.patch_embed = PatchEmbed1D(in_channels, embed_dim, patch_size)

        # Transformer encoder
        self.encoder = ViTEncoder1D(embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio)

        # Decide which transformer layers to use as skips: L evenly spaced indices from 1..depth-1
        if depth <= L:
            take = list(range(1, depth))
            # pad with last index if needed
            while len(take) < L:
                take.append(depth)
        else:
            take = [max(1, (i * depth) // (L + 1)) for i in range(1, L + 1)]
        self.take_indices: List[int] = take

        # Projections from embed_dim to per-level channels for skips and bottleneck
        self.proj_skips = nn.ModuleList([nn.Conv1d(embed_dim, widths[i], kernel_size=1, bias=True) for i in range(0, L)])
        self.proj_bottom = nn.Conv1d(embed_dim, widths[L], kernel_size=1, bias=True)

        # Decoder stages from bottom L -> 1, each outputs widths[j-1]
        self.decoder = nn.ModuleList(
            [
                UpConvUNETR(
                    in_channels=widths[j],
                    skip_channels=(widths[j - 1] if use_skips else 0),
                    out_channels=widths[j - 1],
                    dropout_p=dropout_ps[j - 1],
                )
                for j in range(L, 0, -1)
            ]
        )

        # Output head
        self.output = OutConv(widths[0], out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_len = x.shape[-1]
        # Patch embedding -> (B, E, N)
        tokens = self.patch_embed(x)
        # Transformer encoder returns [h@idx1, ..., h@idxL, h@final], each (B, E, N)
        hiddens = self.encoder(tokens, self.take_indices)

        # Prepare bottom feature from final hidden
        bottom = self.proj_bottom(hiddens[-1])  # (B, C_L, N)

        # Prepare list of skip features for levels 0..L-1
        skip_feats: List[Optional[torch.Tensor]] = []
        if self.use_skips:
            for i in range(len(self.widths) - 1):
                h = hiddens[i] if i < len(hiddens) - 1 else hiddens[-2]
                skip_feats.append(self.proj_skips[i](h))  # (B, C_i, N)
        else:
            skip_feats = [None for _ in range(len(self.widths) - 1)]

        # Decode from bottom to top
        x = bottom
        for idx, dec in enumerate(self.decoder, start=1):
            # pick corresponding skip from high to low levels
            skip = skip_feats[-idx] if self.use_skips else None
            x = dec(x, skip)

        # Always trim/resize back to original length (no conditional on tensor shapes)
        x = nn.functional.interpolate(x, size=in_len, mode="linear", align_corners=False)

        return self.output(x)


__all__ = ["UNETR"]


if __name__ == "__main__":
    # Helper to export a default UNETR to ONNX for Netron
    def export_default_onnx(out_path: str = "UNETR.onnx") -> None:
        in_ch, out_ch = 30, 30
        widths = [16, 32, 64]
        drop_p = [0.2, 0.2, 0.2]
        model = UNETR(in_ch, out_ch, widths=widths, dropout_ps=drop_p, patch_size=8, embed_dim=128, depth=6, num_heads=4, use_skips=True)
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
