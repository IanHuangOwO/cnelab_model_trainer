from __future__ import annotations

from typing import List, Optional

import math
import torch
from torch import nn


class DoubleConv(nn.Module):
    """(Conv1d -> BatchNorm1d -> ReLU -> Dropout1d) x 2."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7, dropout_p: float = 0.2) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
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


def sinusoidal_position_encoding(length: int, dim: int, device: torch.device) -> torch.Tensor:
    """Create standard sinusoidal positional encoding (length, dim)."""
    pe = torch.zeros(length, dim, device=device)
    position = torch.arange(0, length, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class PatchEmbed1D(nn.Module):
    """1D patch embedding via Conv1d with stride=patch_size.

    Input: (B, C_in, T) -> (B, C_embed, T//P)
    """

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class TransformerBlock1D(nn.Module):
    """ViT-like transformer block operating on sequences from 1D patches.

    Expects input as (S, B, C) for PyTorch MultiheadAttention compatibility.
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=False)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (S, B, C)
        h = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = x + h
        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + h
        return x


class ViTEncoder1D(nn.Module):
    """Stack of Transformer blocks with sinusoidal positional encodings.

    Returns hidden states at requested indices for building UNETR skips.
    """

    def __init__(self, embed_dim: int, depth: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock1D(embed_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(depth)]
        )

    def forward(self, x_tokens: torch.Tensor, take_indices: List[int]) -> List[torch.Tensor]:
        # x_tokens: (B, C, N) -> (N, B, C)
        B, C, N = x_tokens.shape
        x = x_tokens.permute(2, 0, 1).contiguous()
        pe = sinusoidal_position_encoding(N, C, x_tokens.device)  # (N, C)
        x = x + pe.unsqueeze(1)  # (N, B, C)

        outs: List[torch.Tensor] = []
        for i, blk in enumerate(self.blocks, start=1):
            x = blk(x)
            if i in take_indices:
                outs.append(x)
        # Also return final x
        outs.append(x)
        # convert each to (B, C, N)
        outs = [t.permute(1, 2, 0).contiguous() for t in outs]
        return outs


class UpConvUNETR(nn.Module):
    """Upsample, optional skip concat, then DoubleConv (channels fused to out_channels)."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, kernel_size: int = 7, dropout_p: float = 0.2) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.conv = DoubleConv(in_channels + (skip_channels if skip_channels > 0 else 0), out_channels, kernel_size, dropout_p)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            # Always resize skip to match x to avoid tracing Python bools
            skip = nn.functional.interpolate(skip, size=x.shape[-1], mode="linear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


__all__ = [
    "DoubleConv",
    "PatchEmbed1D",
    "TransformerBlock1D",
    "ViTEncoder1D",
    "UpConvUNETR",
    "OutConv",
]
