from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import nn


class LayerNormChannel(nn.Module):
    """LayerNorm over channel dim for 1D sequences (B, C, T)."""

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, T, C) -> LN -> back
        x_perm = x.permute(0, 2, 1)
        x_perm = self.ln(x_perm)
        return x_perm.permute(0, 2, 1)


class MLP1D(nn.Module):
    def __init__(self, channels: int, hidden_mult: int = 4) -> None:
        super().__init__()
        hidden = channels * hidden_mult
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv1d(hidden, channels, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WindowMSA1D(nn.Module):
    """Multi-head self-attention within 1D non-overlapping windows."""

    def __init__(self, embed_dim: int, num_heads: int, window_size: int) -> None:
        super().__init__()
        self.window_size = window_size
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        B, C, T = x.shape
        W = self.window_size
        pad_len = (W - (T % W)) % W
        # Always pad (no-op when pad_len==0) to avoid tracing Python bools
        x = nn.functional.pad(x, (0, pad_len))  # pad on time axis
        T_pad = x.shape[-1]
        # reshape into windows
        num_win = T_pad // W
        xw = x.view(B, C, num_win, W).permute(2, 0, 3, 1).contiguous()  # (num_win, B, W, C)
        xw = xw.view(num_win * B, W, C)  # (N*B, W, C)
        # MHA expects (W, N*B, C)
        qkv = xw.permute(1, 0, 2).contiguous()
        out, _ = self.mha(qkv, qkv, qkv, need_weights=False)
        out = out.permute(1, 0, 2).contiguous()  # (N*B, W, C)
        out = out.view(num_win, B, W, C).permute(1, 3, 0, 2).contiguous()  # (B, C, num_win, W)
        out = out.view(B, C, T_pad)
        # Always trim back to original length (no-op when pad_len==0)
        out = out[..., :T]
        return out


class SwinBlock1D(nn.Module):
    """Swin-style block with (shifted) windowed MSA and MLP for 1D signals."""

    def __init__(self, channels: int, num_heads: int, window_size: int, shift: bool, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift = shift
        self.norm1 = LayerNormChannel(channels)
        self.attn = WindowMSA1D(channels, num_heads, window_size)
        self.norm2 = LayerNormChannel(channels)
        self.mlp = MLP1D(channels)
        self.drop = nn.Dropout1d(p=dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        B, C, T = x.shape
        shift_sz = self.window_size // 2 if self.shift else 0

        residual = x
        x = self.norm1(x)
        if shift_sz > 0:
            x = torch.roll(x, shifts=-shift_sz, dims=-1)
        x = self.attn(x)
        if shift_sz > 0:
            x = torch.roll(x, shifts=shift_sz, dims=-1)
        x = x + residual
        
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        x = self.drop(x)
        return x


class PatchMerging1D(nn.Module):
    """Downsample by 2 in time and adjust channels."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.pool(x))


class PatchExpand1D(nn.Module):
    """Upsample by 2 in time and adjust channels."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.up(x))


class SwinStageDown(nn.Module):
    """A stage with N Swin blocks followed by downsample (except at bottom)."""

    def __init__(self, in_channels: int, out_channels: Optional[int], depth: int, num_heads: int, window_size: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [SwinBlock1D(in_channels, num_heads, window_size, shift=(i % 2 == 1), dropout_p=dropout_p) for i in range(depth)]
        )
        self.down = PatchMerging1D(in_channels, out_channels) if out_channels is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        if self.down is not None:
            x = self.down(x)
        return x


class SwinStageUp(nn.Module):
    """Upsample, optional skip concat + fuse, then N Swin blocks."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, depth: int, num_heads: int, window_size: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.up = PatchExpand1D(in_channels, out_channels)
        fuse_in = out_channels + (skip_channels if skip_channels > 0 else 0)
        self.fuse = nn.Conv1d(fuse_in, out_channels, kernel_size=1, bias=True)
        self.blocks = nn.ModuleList(
            [SwinBlock1D(out_channels, num_heads, window_size, shift=(i % 2 == 1), dropout_p=dropout_p) for i in range(depth)]
        )

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            x = torch.cat([skip, x], dim=1)
        x = self.fuse(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class InputProj(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7) -> None:
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=pad, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7) -> None:
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=pad, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


__all__ = [
    "LayerNormChannel",
    "MLP1D",
    "WindowMSA1D",
    "SwinBlock1D",
    "PatchMerging1D",
    "PatchExpand1D",
    "SwinStageDown",
    "SwinStageUp",
    "InputProj",
    "OutConv",
]
