from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


def _to_btC(t: torch.Tensor) -> torch.Tensor:
    # Convert (B, C, T) -> (B, T, C); leave (B, T, C) as-is
    if t.dim() == 3 and t.size(1) < t.size(2):
        return t.permute(0, 2, 1)
    return t


def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    # x: (B, T, C); mask: (B, T) bool
    if mask is None:
        return x.mean()
    m = mask.unsqueeze(-1).expand_as(x)
    sel = x.masked_select(m)
    return sel.mean() if sel.numel() > 0 else x.new_tensor(0.0)


@dataclass
class LossResult:
    loss: torch.Tensor
    logs: Dict[str, float]
    norm: int


class LossComputer:
    """Simple configurable loss + basic metrics.

    cfg['train']['loss']:
      type: mse | mae (default: mse)
      zscore: true|false (default: true)
    """

    def __init__(self, cfg: dict) -> None:
        spec = (cfg.get("train", {}).get("loss", {}) or {}) if isinstance(cfg, dict) else {}
        self.kind = str(spec.get("type", "mse")).lower()
        self.zscore = bool(spec.get("zscore", True))
        if self.kind not in {"mse", "mae"}:
            self.kind = "mse"
        self.eps = 1e-10

    def __call__(self, out: torch.Tensor, *, target: torch.Tensor, keep_mask: Optional[torch.Tensor]) -> LossResult:
        # Shapes to (B, T, C)
        y = _to_btC(target)
        x = _to_btC(out)

        if self.zscore:
            xm, xs = x.mean(dim=1, keepdim=True), x.std(dim=1, keepdim=True)
            ym, ys = y.mean(dim=1, keepdim=True), y.std(dim=1, keepdim=True)
            x = (x - xm) / (xs + self.eps)
            y = (y - ym) / (ys + self.eps)

        if self.kind == "mae":
            main = _masked_mean((x - y).abs(), keep_mask)
        else:
            main = _masked_mean((x - y).pow(2), keep_mask)

        # Metrics (without z-score) for reference
        x_raw, y_raw = _to_btC(out), _to_btC(target)
        mse_val = float(_masked_mean((x_raw - y_raw).pow(2), keep_mask).detach().cpu())
        mae_val = float(_masked_mean((x_raw - y_raw).abs(), keep_mask).detach().cpu())
        logs = {"loss": float(main.detach().cpu()), "mse": mse_val, "mae": mae_val}

        # Norm: approx number of valid elements
        if keep_mask is not None:
            norm = int(keep_mask.sum().item() * y.size(-1))
        else:
            norm = int(y.numel())
        return LossResult(main, logs, norm)

