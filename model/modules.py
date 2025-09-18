"""Generic building blocks used by models in this repository.

This module keeps reusable, framework-agnostic layers:

- ExpandConv1x1: 1x1 convolution to project channel dimension (c -> d) per time step.
- PositionalEmbedding: learned or sinusoidal positional encodings added to inputs.
- ScaledDotProductAttention: core attention primitive with masking and dropout.
- MultiHeadAttention: standard multi-head attention over sequences.
- FeedForward: position-wise MLP (two linear layers with activation and dropout).

All layers expect PyTorch tensors and favor clear shape semantics documented
in each class. These blocks are intentionally generic so they can be composed
into different architectures (e.g., ART in model/ART.py).
"""

import math
from typing import Optional

import torch
from torch import nn, Tensor


class ExpandConv1x1(nn.Module):
    """1x1 Conv channel projector: (B, C, T) -> (B, T, D).

    Args:
        in_channels: input channel count C.
        out_channels: model dim D after projection.
        bias: whether Conv1d uses bias.
        activation: optional activation (str or nn.Module), e.g., 'relu' or 'gelu'.
        dropout: dropout probability applied after activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        bias: bool = True,
        activation: str | nn.Module | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias)
        if isinstance(activation, str):
            act_cls = getattr(nn, activation.capitalize(), None)
            if act_cls is None:
                if activation.lower() == "gelu":
                    self.act = nn.GELU()
                elif activation.lower() == "relu":
                    self.act = nn.ReLU()
                else:
                    raise ValueError(f"Unsupported activation: {activation}")
            else:
                self.act = act_cls()
        else:
            self.act = activation
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, T) -> (B, D, T) -> (B, T, D)
        x = self.proj(x)
        if self.act is not None:
            x = self.act(x)
        x = self.drop(x)
        return x.permute(0, 2, 1)


class PositionalEmbedding(nn.Module):
    """Add positional information to a sequence.

    Supports either learned embeddings or fixed sinusoidal encodings.

    Args:
        max_len: Maximum sequence length supported by the buffer/table.
        d_model: Embedding dimension (last dim of input/output).
        mode: 'learned' or 'sinusoidal'.

    Input/Output:
        x: (B, S, d_model) → returns (B, S, d_model) with positions added.
    """

    def __init__(self, max_len: int, d_model: int, mode: str = "learned") -> None:
        super().__init__()
        if mode not in {"learned", "sinusoidal"}:
            raise ValueError("mode must be 'learned' or 'sinusoidal'")
        self.max_len = int(max_len)
        self.d_model = int(d_model)
        self.mode = mode

        if self.mode == "learned":
            self.table = nn.Embedding(self.max_len, self.d_model)
        else:
            pe = torch.zeros(self.max_len, self.d_model)
            position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.d_model, 2, dtype=torch.float)
                * (-(math.log(10000.0) / max(1, (self.d_model - 1))))
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, seq_len, d_model)
        b, s, _ = x.shape
        if self.mode == "learned":
            positions = torch.arange(s, device=x.device)
            pos_emb = self.table(positions)  # (seq, d_model)
        else:
            pos_emb = self.pe[:s, :]
        return x + pos_emb.unsqueeze(0)


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention with optional mask and dropout.

    Mask semantics (broadcast to (..., Q, K)):
      - Boolean: True means masked/excluded (logits set to -inf).
      - Float   : Additive mask added to logits (0 keep, -inf mask).
    """

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        d_k = q.size(-1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask
        attn = scores.softmax(dim=-1)
        attn = self.drop(attn)
        return attn @ v


class MultiHeadAttention(nn.Module):
    """Multi-head attention over sequences.

    Input/Output shapes:
      - q, k, v: (B, S, d_model)
      - attn_mask: broadcastable to (B, H, S_q, S_k) with True = mask
      - returns: (B, S, d_model)
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model, bias=True)
        self.wk = nn.Linear(d_model, d_model, bias=True)
        self.wv = nn.Linear(d_model, d_model, bias=True)
        self.attn = ScaledDotProductAttention(dropout)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        b, s, _ = x.shape
        x = x.view(b, s, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x: Tensor) -> Tensor:
        b, h, s, d = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(b, s, h * d)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        q = self._split_heads(self.wq(q))
        k = self._split_heads(self.wk(k))
        v = self._split_heads(self.wv(v))

        if attn_mask is not None:
            # Broadcast masks to (B, H, Q, K)
            if attn_mask.dim() == 4 and attn_mask.size(1) == 1:
                attn_mask = attn_mask.expand(-1, self.num_heads, -1, -1)

        context = self.attn(q, k, v, attn_mask=attn_mask)
        context = self._merge_heads(context)
        out = self.proj(context)
        out = self.drop(out)
        return out


class FeedForward(nn.Module):
    """Position-wise MLP (two linear layers with activation and dropout)."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        activation: str | nn.Module = "relu",
    ) -> None:
        super().__init__()
        if isinstance(activation, str):
            act = getattr(nn, activation.capitalize(), None)
            if act is None:
                # fall back to functional gelu via nn.GELU
                if activation.lower() == "gelu":
                    act_layer = nn.GELU()
                elif activation.lower() == "relu":
                    act_layer = nn.ReLU()
                else:
                    raise ValueError(f"Unsupported activation: {activation}")
            else:
                act_layer = act()
        else:
            act_layer = activation

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            act_layer,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


__all__ = [
    "PositionalEmbedding",
    "ScaledDotProductAttention",
    "MultiHeadAttention",
    "FeedForward",
    "ExpandConv1x1",
]
