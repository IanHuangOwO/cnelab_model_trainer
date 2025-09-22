from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch


def _make_padding_mask_channel(x: torch.Tensor, pad: float = 0.0) -> torch.Tensor:
    return (x[:, 0, :] != pad)


def _make_causal_mask(T: int, device=None) -> torch.Tensor:
    i = torch.arange(T, device=device)
    return i.unsqueeze(0) >= i.unsqueeze(1)


@dataclass(frozen=True)
class CollatedBatch:
    src: torch.Tensor        # (B, C, T)
    trg: torch.Tensor        # (B, C, T-1) — decoder inputs and loss target
    src_mask: torch.Tensor   # (B, T) bool
    trg_mask: torch.Tensor   # (B, T-1, T-1) bool (padding + causal)
    ntokens: int             # number of valid target tokens

    @staticmethod
    def from_tensors(src: torch.Tensor, tgt: torch.Tensor, pad: float = 0.0) -> "CollatedBatch":
        # Teacher forcing alignment: model outputs length T-1
        trg = tgt[:, :, :-1]

        # Masks derived from channel 0 only
        src_mask = _make_padding_mask_channel(src, pad=pad)  # (B, T)
        trg_keep = _make_padding_mask_channel(trg, pad=pad)  # (B, T-1)
        causal = _make_causal_mask(trg.size(-1), device=trg.device)  # (T-1, T-1)
        trg_mask = trg_keep.unsqueeze(-1) & trg_keep.unsqueeze(-2) & causal

        ntokens = int(trg.ne(pad).sum().item())
        return CollatedBatch(src=src, trg=trg, src_mask=src_mask, trg_mask=trg_mask, ntokens=ntokens)

    def to(self, device: torch.device) -> "CollatedBatch":
        return CollatedBatch(
            src=self.src.to(device),
            trg=self.trg.to(device),
            src_mask=self.src_mask.to(device),
            trg_mask=self.trg_mask.to(device),
            ntokens=self.ntokens,
        )


def collate_eeg_batch_channel(samples: List[Tuple[torch.Tensor, torch.Tensor, dict]], pad: float = 0.0) -> CollatedBatch:
    """Collate that stacks (attr, target, meta) and builds masks from channel 0.

    samples: list of (attr, target, meta), each shaped (C, T).
    Returns a Batch compatible with the training loop.
    """
    attrs, targets, _ = zip(*samples)
    src = torch.stack(list(attrs), dim=0)
    tgt = torch.stack(list(targets), dim=0)
    return CollatedBatch.from_tensors(src, tgt, pad=pad)