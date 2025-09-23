from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from model.ART_blocks import (
    ExpandConv1x1,
    PositionalEmbedding,
    MultiHeadAttention,
    FeedForward,
)

class TransformerEncoderBlock(nn.Module):
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
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        return self.norm(x)


class TransformerDecoderBlock(nn.Module):
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
        h = self.self_mha(x, x, x, attn_mask=cross_attn_mask)
        x = self.ln1(x + self.drop1(h))

        h = self.cross_mha(x, memory, memory, attn_mask=self_attn_mask)
        x = self.ln2(x + self.drop2(h))

        h = self.ffn(x)
        x = self.ln3(x + self.drop3(h))
        return x


class TransformerDecoder(nn.Module):
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
        for layer in self.layers:
            x = layer(x, memory, self_attn_mask=self_attn_mask, cross_attn_mask=cross_attn_mask)
        return self.norm(x)


class Reconstructor(nn.Module):
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

class ArtifactRemovalTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_size: int = 128,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        feedforward_size: int = 2048,
        dropout: float = 0.1,
        max_len: int = 2048,
        pos_mode: str = "sinusoidal",
        recon_log_softmax: bool = False,
        recon_zscore: str | None = None,
    ) -> None:
        super().__init__()
        
        self.src_embed = nn.Sequential(
            ExpandConv1x1(in_channels, embedding_size),
            PositionalEmbedding(max_len=max_len, d_model=embedding_size, mode=pos_mode),
            nn.Dropout(dropout),
        )

        self.encoder = TransformerEncoder(
            d_model=embedding_size,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            d_ff=feedforward_size,
            dropout=dropout,
            attn_dropout=dropout,
        )
        
        self.tgt_embed = nn.Sequential(
            ExpandConv1x1(out_channels, embedding_size),
            PositionalEmbedding(max_len=max_len, d_model=embedding_size, mode=pos_mode),
            nn.Dropout(dropout),
        )
        self.decoder = TransformerDecoder(
            d_model=embedding_size,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            d_ff=feedforward_size,
            dropout=dropout,
            attn_dropout=dropout,
        )
        
        self.reconstructor = Reconstructor(
            d_model=embedding_size,
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
        src_x = self.src_embed(src)

        memory = self.encoder(src_x, attn_mask=src_mask)
        
        tgt_x = self.tgt_embed(tgt)

        out = self.decoder(tgt_x, memory, src_mask, tgt_mask)
        
        return self.reconstructor(out)


def build_model_from_config(cfg: dict) -> ArtifactRemovalTransformer:
    """Build ArtifactRemovalTransformer from a loaded config dict.

    Accepts either the full config with a top-level "model" section or a
    dict that is itself the model section.
    """
    m = cfg.get("model", cfg) if isinstance(cfg, dict) else {}

    in_channels = int(m.get("in_channels", 30))
    out_channels = int(m.get("out_channels", 30))

    # Widths
    embedding_size = int(m.get("embedding_size", 128))
    feedforward_size = int(m.get("feedforward_size", 2048))

    # Depths
    num_layers = int(m.get("num_layers", 6))
    num_encoder_layers = int(m.get("num_encoder_layers", num_layers))
    num_decoder_layers = int(m.get("num_decoder_layers", num_layers))

    # Attention/regularization
    num_heads = int(m.get("num_heads", 8))
    dropout = float(m.get("dropout", 0.1))
    max_len = int(m.get("max_len", 2048))
    pos_mode = str(m.get("pos_mode", "sinusoidal"))

    # Reconstruction head options
    recon_log_softmax = bool(m.get("recon_log_softmax", False))
    recon_zscore = m.get("recon_zscore", None)

    return ArtifactRemovalTransformer(
        in_channels=in_channels,
        out_channels=out_channels,
        embedding_size=embedding_size,
        feedforward_size=feedforward_size,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        num_heads=num_heads,
        dropout=dropout,
        max_len=max_len,
        pos_mode=pos_mode,
        recon_log_softmax=recon_log_softmax,
        recon_zscore=recon_zscore,
    )

__all__ = [
    "ArtifactRemovalTransformer",
    "build_model_from_config"
]
