from __future__ import annotations

"""Training utilities: Noam learning-rate schedule and helpers.

This module provides a drop-in replacement for the original NoamOpt
wrapper used in the codebase, plus a LambdaLR-compatible helper.
"""

from typing import Optional, Callable

import torch


class Noam:
    """Transformer (Noam) learning-rate schedule wrapper.

    lr(step) = factor * d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)
    """

    def __init__(self, d_model: int, factor: float, warmup: int, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self.d_model = float(d_model)
        self.factor = float(factor)
        self.warmup = int(warmup)
        self._step = 0
        self._lr = 0.0

    @property
    def lr(self) -> float:
        return self._lr

    def rate(self, step: Optional[int] = None) -> float:
        s = self._step if step is None else int(step)
        s = max(1, s)
        return self.factor * (self.d_model ** -0.5) * min(s ** -0.5, s * (self.warmup ** -1.5))

    def step(self) -> None:
        self._step += 1
        lr = self.rate()
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        self._lr = lr
        self.optimizer.step()

    def zero_grad(self, set_to_none: bool = False) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {
            "d_model": self.d_model,
            "factor": self.factor,
            "warmup": self.warmup,
            "_step": self._step,
            "_lr": self._lr,
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.d_model = float(state["d_model"])  # type: ignore[assignment]
        self.factor = float(state["factor"])    # type: ignore[assignment]
        self.warmup = int(state["warmup"])     # type: ignore[assignment]
        self._step = int(state["_step"])       # type: ignore[assignment]
        self._lr = float(state["_lr"])         # type: ignore[assignment]
        self.optimizer.load_state_dict(state["optimizer"])


## Note: NoamOpt removed in favor of Noam (above) and make_noam_lambda helper.


def make_noam_lambda(model_size: int, factor: float, warmup: int) -> Callable[[int], float]:
    """Return a LambdaLR scheduler function implementing the Noam schedule.

    Example:
        opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=make_noam_lambda(d_model, 1.0, 4000))
        # then call sched.step() after each optimizer.step()
    """
    model_size = int(model_size)
    factor = float(factor)
    warmup = int(warmup)

    def _fn(step: int) -> float:
        step = max(1, int(step))
        return factor * ((model_size ** -0.5) * min(step ** -0.5, step * (warmup ** -1.5)))

    return _fn


def _infer_d_model(model: torch.nn.Module) -> int:
    # Direct attribute is best
    if hasattr(model, "embedding_size"):
        return int(getattr(model, "embedding_size"))
    if hasattr(model, "d_model"):
        return int(getattr(model, "d_model"))
    # Try common patterns: src_embed submodules exposing d_model
    try:
        emb = getattr(model, "src_embed")
        # Sequential: try to find first submodule with d_model
        if hasattr(emb, "_modules"):
            for m in emb._modules.values():
                if hasattr(m, "d_model"):
                    return int(getattr(m, "d_model"))
        # Index access fallback
        if hasattr(emb, "__getitem__") and hasattr(emb[1], "d_model"):
            return int(emb[1].d_model)
    except Exception:
        pass
    # Fallback default
    return 128


def build_noam_from_config(
    cfg: dict,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Noam:
    """Construct a Noam wrapper from a config dict and a model.

    Looks under cfg['train'] for optimizer/scheduler settings.
    """
    train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}
    opt_cfg = train_cfg.get("optimizer", {}) if isinstance(train_cfg, dict) else {}
    sch_cfg = train_cfg.get("scheduler", {}) if isinstance(train_cfg, dict) else {}

    # Params
    d_model = _infer_d_model(model)
    factor = float(sch_cfg.get("factor", 1.0))
    warmup = int(sch_cfg.get("warmup", 400))

    if optimizer is None:
        # Build base optimizer with defaults
        lr = float(train_cfg.get("lr", 0.0))
        betas = tuple(opt_cfg.get("betas", (0.9, 0.98)))  # type: ignore[assignment]
        eps = float(opt_cfg.get("eps", 1e-9))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps)

    return Noam(d_model, factor, warmup, optimizer)
