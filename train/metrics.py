from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional

import math
import torch


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


def mse(x: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    x, y = _to_btC(x), _to_btC(y)
    return float(_masked_mean((x - y).pow(2), mask).detach().cpu())


def mae(x: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    x, y = _to_btC(x), _to_btC(y)
    return float(_masked_mean((x - y).abs(), mask).detach().cpu())


def rmse(x: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    return math.sqrt(max(mse(x, y, mask), 0.0))


def corr(x: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    # Pearson correlation across all selected elements
    x, y = _to_btC(x), _to_btC(y)
    if mask is not None:
        m = mask.unsqueeze(-1).expand_as(x)
        x = x.masked_select(m)
        y = y.masked_select(m)
    x = x - x.mean()
    y = y - y.mean()
    denom = (x.std() + 1e-10) * (y.std() + 1e-10)
    if float(denom) == 0.0:
        return 0.0
    return float((x * y).mean().detach().cpu() / denom)


def r2(x: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    # Coefficient of determination across selected elements
    x, y = _to_btC(x), _to_btC(y)
    if mask is not None:
        m = mask.unsqueeze(-1).expand_as(x)
        x = x.masked_select(m)
        y = y.masked_select(m)
    ss_res = torch.sum((y - x) ** 2)
    ss_tot = torch.sum((y - y.mean()) ** 2) + 1e-10
    return float(1.0 - (ss_res / ss_tot).detach().cpu())


_REGISTRY: Dict[str, Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], float]] = {
    "mse": mse,
    "mae": mae,
    "rmse": rmse,
    "corr": corr,
    "r2": r2,
}


class MetricsComputer:
    """Compute a configurable set of metrics.

    cfg['train']['metrics']: list of metric names. Defaults to ['mse', 'mae'].
    """

    def __init__(self, cfg: dict, defaults: Iterable[str] = ("mse", "mae")) -> None:
        names = cfg.get("train", {}).get("metrics") if isinstance(cfg, dict) else None
        if not names:
            names = list(defaults)
        self.names: list[str] = [str(n).lower() for n in names if str(n).lower() in _REGISTRY]
        if not self.names:
            self.names = list(defaults)

    def __call__(self, out: torch.Tensor, *, target: torch.Tensor, keep_mask: Optional[torch.Tensor]) -> Dict[str, float]:
        results: Dict[str, float] = {}
        for name in self.names:
            fn = _REGISTRY[name]
            results[name] = fn(out, target, keep_mask)
        return results

