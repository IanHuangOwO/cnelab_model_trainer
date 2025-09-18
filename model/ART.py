"""ART: Artifact Removal Transformer (encoder–decoder, non‑autoregressive).

This module implements the EEG Artifact Removal Transformer (ART) described
in the provided paper illustration (Figure 1C). The architecture consists of:

- ExpandConv (1x1 conv) to project channel space `c -> τ` per time step.
- Positional encoding added to the `τ`-dim embeddings without changing time T.
- Transformer Encoder: L blocks, each with MHA then FeedForward, using
  Post-LN Add&Norm around each sublayer.
- Transformer Decoder: L blocks with self-attn, cross-attn, and FeedForward,
  also using Post-LN Add&Norm. Decoding is non‑autoregressive (no causal mask).
- Reconstructor (outside forward): a linear layer `generator: τ -> c` exposed
  as an attribute to map the model features back to channel space. Z-score
  normalization and any LogSoftmax for stabilization is handled in the training
  pipeline (see EEGART/tf_loss.py), not inside this module.

Tensors follow these conventions:
- Source `src`: (B, C_src, T)
- Target `tgt`: (B, C_tgt, T)
- Embedded sequences: (B, T, D) where D = τ = d_model
- Attention masks use boolean semantics where True masks/excludes positions
  (we disable decoder causal masking, but accept optional encoder padding mask).
"""

from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .modules import (
    ExpandConv1x1,
    PositionalEmbedding,
    MultiHeadAttention,
    FeedForward,
)


class ArtifactRemovalTransformer(nn.Module):
    """Artifact Removal Transformer (ART) for EEG.

    Pipeline:
      1) ExpandConv1x1 projects channels `c -> τ` at each time step.
      2) PositionalEncoding adds sine/cos embeddings.
      3) TransformerEncoder produces memory features of shape (B, T, τ).
      4) TransformerDecoder consumes a target sequence (non‑autoregressive,
         no causal mask) and cross‑attends to the encoder memory.
      5) `generator` is a linear reconstructor τ -> c used outside forward.

    This module is designed to match your training loop:
      - Exposes `src_embed[0].d_model` for NoamOpt scheduling.
      - Exposes `generator` used by loss to produce channel outputs.

    Args:
        in_channels: Source EEG channels `c`.
        out_channels: Target channels to reconstruct (usually equals `c`).
        d_model: Embedding dimension τ.
        num_encoder_layers: Number of encoder blocks L.
        num_decoder_layers: Number of decoder blocks.
        num_heads: Attention heads h.
        d_ff: Feed-forward hidden size τ'.
        dropout: Dropout probability in projections and residual paths.
        max_len: Max supported sequence length for positional embeddings.
        pos_mode: 'sinusoidal' or 'learned' positional embedding mode.
        recon_log_softmax: If True, apply LogSoftmax in the reconstructor.
        recon_zscore: z-score mode in reconstructor: None, "batch", or "time".

    Shapes:
        src: (B, in_channels, T)
        tgt: (B, out_channels, T)
        Returns: decoder features (B, T, d_model). Use `self.generator` to map
                 to (B, T, out_channels) when computing loss.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        d_model: int = 128,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 2048,
        pos_mode: str = "sinusoidal",
        recon_log_softmax: bool = False,
        recon_zscore: str | None = None,
    ) -> None:
        super().__init__()
        
        self.src_embed = nn.Sequential(
            ExpandConv1x1(in_channels, d_model),
            PositionalEmbedding(max_len=max_len, d_model=d_model, mode=pos_mode),
            nn.Dropout(dropout),
        )

        self.encoder = TransformerEncoder(
            d_model=d_model,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            attn_dropout=dropout,
        )
        
        self.tgt_embed = nn.Sequential(
            ExpandConv1x1(out_channels, d_model),
            PositionalEmbedding(max_len=max_len, d_model=d_model, mode=pos_mode),
            nn.Dropout(dropout),
        )
        self.decoder = TransformerDecoder(
            d_model=d_model,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            attn_dropout=dropout,
        )
        
        self.reconstructor = Reconstructor(
            d_model=d_model,
            out_channels=out_channels,
            log_softmax=recon_log_softmax,
            zscore=recon_zscore,
        )

    def forward(
        self,
        src: Tensor,
        tgt: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass of ART.

        Non‑autoregressive decoding: no causal mask is applied in the decoder.

        Args:
            src: (B, C_src, T) raw/noisy EEG
            tgt: (B, C_tgt, T) target conditioning sequence (e.g., noisy, zeros,
                 or pseudo‑clean), must be provided since decoder is always used
            src_mask: Optional padding keep‑mask (B, 1, T) or (B, T)
            tgt_mask: Ignored for causality; you may still supply padding masks
            apply_reconstructor: If True, apply the reconstructor head and
                return channel outputs (B, T, out_channels). If False, return
                decoder features (B, T, d_model).

        Returns:
            Tensor of shape (B, T, d_model) if apply_reconstructor is False,
            otherwise (B, T, out_channels).
        """
        # Encode source: (B, C_src, T) -> (B, S, d_model)
        enc = self.src_embed(src)

        # Normalize src_mask to boolean True=mask with shape (B, 1, 1, S)
        enc_attn_mask = None
        if src_mask is not None:
            if src_mask.dtype == torch.bool:
                m = src_mask
            else:
                m = src_mask != 0
            if m.dim() == 3:  # (B, 1, S)
                enc_attn_mask = (~m).unsqueeze(1)
            elif m.dim() == 2:  # (B, S)
                enc_attn_mask = (~m).unsqueeze(1).unsqueeze(1)
            else:
                enc_attn_mask = ~m

        memory = self.encoder(enc, attn_mask=enc_attn_mask)
        
        dec_inp = self.tgt_embed(tgt)

        # Non-autoregressive: disable causal/self mask for decoder
        self_attn_mask = None

        cross_attn_mask = enc_attn_mask
        out = self.decoder(dec_inp, memory, self_attn_mask, cross_attn_mask)
        
        return self.reconstructor(out)


class TransformerEncoderBlock(nn.Module):
    """Encoder block with Post‑LN Add&Norm.

    Sequence of sublayers per block:
      1) Multi-Head Self-Attention
      2) Residual add + LayerNorm
      3) Position-wise FeedForward (ReLU by default)
      4) Residual add + LayerNorm

    Args:
        d_model: Model dimension τ.
        num_heads: Number of attention heads h (τ must be divisible by h).
        d_ff: Hidden size of the feed-forward network τ' (expansion factor).
        dropout: Dropout probability after attention and FFN.
        attn_dropout: Dropout probability inside attention weights.

    Shapes:
        x: (B, T, d_model)
        attn_mask: Optional bool mask broadcastable to (B, H, T, T), True = mask.
        Returns: (B, T, d_model)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout=attn_dropout)
        self.drop1 = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model, eps=1e-5)

        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)
        self.drop2 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        h = self.mha(x, x, x, attn_mask=attn_mask)
        x = self.ln1(x + self.drop1(h))
        
        h = self.ffn(x)
        x = self.ln2(x + self.drop2(h))
        return x


class TransformerDecoderBlock(nn.Module):
    """Decoder block with self- and cross-attention (Post‑LN Add&Norm).

    Sequence per block:
      1) Self-attention on decoder inputs (no causal mask used here).
      2) Residual add + LayerNorm
      3) Cross-attention over encoder memory
      4) Residual add + LayerNorm
      5) Position-wise FeedForward
      6) Residual add + LayerNorm

    Args are identical to TransformerEncoderBlock.

    Shapes:
        x: (B, T, d_model)
        memory: (B, T, d_model) from encoder
        self_attn_mask: Optional bool mask (we expect None for non‑AR decoding)
        cross_attn_mask: Optional bool mask for encoder padding
        Returns: (B, T, d_model)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=1e-5)
        self.self_mha = MultiHeadAttention(d_model, num_heads, dropout=attn_dropout)
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model, eps=1e-5)
        self.cross_mha = MultiHeadAttention(d_model, num_heads, dropout=attn_dropout)
        self.drop2 = nn.Dropout(dropout)

        self.ln3 = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        self_attn_mask: Optional[Tensor] = None,
        cross_attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        h = self.self_mha(x, x, x, attn_mask=self_attn_mask)
        x = self.ln1(x + self.drop1(h))

        h = self.cross_mha(x, memory, memory, attn_mask=cross_attn_mask)
        x = self.ln2(x + self.drop2(h))

        h = self.ffn(x)
        x = self.ln3(x + self.drop3(h))
        return x


class TransformerDecoder(nn.Module):
    """Stack of decoder blocks with final LayerNorm.

    Args:
        d_model, num_layers, num_heads, d_ff, dropout, attn_dropout: see blocks.
    """
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        self_attn_mask: Optional[Tensor] = None,
        cross_attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply L decoder blocks and a final norm.

        Args:
            x: (B, T, d_model) decoder inputs
            memory: (B, T, d_model) encoder outputs
            self_attn_mask: Optional bool mask, True = masked
            cross_attn_mask: Optional bool mask for encoder padding
        Returns:
            (B, T, d_model)
        """
        for layer in self.layers:
            x = layer(x, memory, self_attn_mask=self_attn_mask, cross_attn_mask=cross_attn_mask)
        return self.norm(x)


class TransformerEncoder(nn.Module):
    """Stack of encoder blocks with final LayerNorm.

    Args:
        d_model, num_layers, num_heads, d_ff, dropout, attn_dropout: see blocks.
    """
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        """Apply L encoder blocks and a final norm.

        Args:
            x: (B, T, d_model)
            attn_mask: Optional bool mask broadcastable to (B, H, T, T)
        Returns:
            (B, T, d_model)
        """
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        return self.norm(x)


class Reconstructor(nn.Module):
    """Reconstruct signals from features with optional stabilization.

    Pipeline: Linear τ->c, optional LogSoftmax along channel dim, optional
    z-score normalization.

    Args:
        d_model: Feature dimension τ.
        out_channels: Output channel dimension c.
        log_softmax: If True, apply LogSoftmax over channel dim after linear.
        zscore: Optional normalization mode:
            - None: no z-score
            - "batch": z-score across batch (dim=0), keeps (T, C) stats
            - "time": per-sample z-score across time (dim=1)
        eps: numerical stability constant.
    """

    def __init__(
        self,
        d_model: int,
        out_channels: int,
        *,
        log_softmax: bool = False,
        zscore: str | None = None,
        eps: float = 1e-10,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, out_channels)
        self.use_log_softmax = log_softmax
        self.zscore = zscore
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, d_model) -> (B, T, C)
        y = self.proj(x)
        if self.use_log_softmax:
            y = F.log_softmax(y, dim=-1)
        if self.zscore is None:
            return y
        if self.zscore == "batch":
            # Normalize across batch dimension, preserving (T, C) statistics
            mean = y.mean(dim=0, keepdim=True)
            std = y.std(dim=0, keepdim=True)
        elif self.zscore == "time":
            # Per-sample z-score across time axis
            mean = y.mean(dim=1, keepdim=True)
            std = y.std(dim=1, keepdim=True)
        else:
            raise ValueError(f"Unsupported zscore mode: {self.zscore}")
        return (y - mean) / (std + self.eps)


__all__ = [
    "ArtifactRemovalTransformer",
]
