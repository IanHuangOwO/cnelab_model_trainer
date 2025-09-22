from __future__ import annotations

import os
import random
from typing import Callable, Optional, Tuple, Dict, List

import torch
from torch.utils.data import Dataset
from IO.reader import read_eeg, gen_eeg


class EEGDataset(Dataset):
    """Simplified EEG dataset driven by a YAML config.

    Config (YAML):
      data:
        root: ./Real_EEG
        # Optional inline split lists. If omitted, lists files under Brain/.
        splits:
          train: ["S001.csv", ...]
          val:   ["S101.csv", ...]
          test:  ["S201.csv", ...]

    Directory layout:
      <root>/<mode>/<Category>/*.csv, where Category in
        {Brain, ChannelNoise, Eye, Heart, LineNoise, Muscle, Other}

    __getitem__ returns (attr, target, meta) as torch.float32 tensors
    of shape [C, T] and a small metadata dict.
    """

    def __init__(
        self,
        *,
        config: Dict,
        mode: str = "train",
        transform: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
        target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        cfg = config
        data_cfg = cfg.get("data", cfg)
        root = data_cfg.get("root")
        if not root:
            raise ValueError("Config must specify data.root")

        self.root = root
        self.split = mode
        self.transform = transform
        self.target_transform = target_transform
        self.rng = random.Random(seed)

        self.categories = [
            "Brain",
            "ChannelNoise",
            "Eye",
            "Heart",
            "LineNoise",
            "Muscle",
            "Other",
        ]

        self.base_dir = os.path.join(self.root, self.split)
        self.brain_dir = os.path.join(self.base_dir, "Brain")
        if not os.path.isdir(self.brain_dir):
            raise FileNotFoundError(f"Brain directory not found: {self.brain_dir}")

        # Inline split list if present; else list files from Brain/
        inline = data_cfg.get("splits") or cfg.get("splits")
        if isinstance(inline, dict) and mode in inline:
            self.files: List[str] = list(inline[mode])
        else:
            self.files = sorted(os.listdir(self.brain_dir))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        fname = self.files[index]
        brain_path = os.path.join(self.brain_dir, fname)

        # Randomly pick a category; fall back to Brain if file missing
        category = self.rng.choice(self.categories)
        noise_path = os.path.join(self.base_dir, category, fname)

        target_np = read_eeg(brain_path)
        if os.path.isfile(noise_path):
            attr_np = read_eeg(noise_path)
        else:
            attr_np = target_np
            category = "Brain"

        target = torch.from_numpy(target_np).to(torch.float32)
        attr = torch.from_numpy(attr_np).to(torch.float32)

        if self.transform is not None:
            attr, target = self.transform(attr, target)
        if self.target_transform is not None:
            target = self.target_transform(target)

        meta: Dict[str, str] = {
            "filename": fname,
            "category": category,
            "split": self.split,
        }
        return attr, target, meta
    
    # No internal config loading; pass config dict to the constructor


class GenEEGDataset(Dataset):
    """Synthetic EEG dataset that generates (attr, target) via gen_eeg.

    Reads parameters from the same YAML config under:
      data:
        splits:
          <mode>:
            C: 30
            T: 1024
            sample_rate: 256.0
            length: 1000          # number of samples to generate
            target: { mode: sine,  noise_std: 0.0, num_components: 3 }
            attr:   { mode: mixed, noise_std: 0.1, num_components: 3 }

    If the block for <mode> is missing, sensible defaults are used.
    """

    def __init__(
        self,
        *,
        config: Dict,
        mode: str = "train",
        transform: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
        target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        cfg = config
        data_cfg = cfg.get("data", cfg)
        gen_cfg = (data_cfg.get("splits") or cfg.get("splits") or {})
        split_cfg: Dict = gen_cfg.get(mode, {}) if isinstance(gen_cfg, dict) else {}

        self.C = int(split_cfg.get("C", 30))
        self.T = int(split_cfg.get("T", 1024))
        self.sample_rate = float(split_cfg.get("sample_rate", 256.0))
        self.length = int(split_cfg.get("length", 1000))

        # Per-stream specs
        self.spec_target: Dict = {
            "mode": split_cfg.get("target", {}).get("mode", "sine"),
            "noise_std": float(split_cfg.get("target", {}).get("noise_std", 0.0)),
            "num_components": int(split_cfg.get("target", {}).get("num_components", 3)),
        }
        self.spec_attr: Dict = {
            "mode": split_cfg.get("attr", {}).get("mode", "mixed"),
            "noise_std": float(split_cfg.get("attr", {}).get("noise_std", 0.1)),
            "num_components": int(split_cfg.get("attr", {}).get("num_components", 3)),
        }

        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        # Derive deterministic per-sample seeds if a base seed is provided
        s_attr = None if self.seed is None else (self.seed * 100003 + index)
        s_tgt = None if self.seed is None else (self.seed * 100019 + index)

        target_np = gen_eeg(
            C=self.C,
            T=self.T,
            sample_rate=self.sample_rate,
            mode=self.spec_target["mode"],
            noise_std=self.spec_target["noise_std"],
            num_components=self.spec_target["num_components"],
            seed=s_tgt,
        )

        attr_np = gen_eeg(
            C=self.C,
            T=self.T,
            sample_rate=self.sample_rate,
            mode=self.spec_attr["mode"],
            noise_std=self.spec_attr["noise_std"],
            num_components=self.spec_attr["num_components"],
            seed=s_attr,
        )

        target = torch.from_numpy(target_np).to(torch.float32)
        attr = torch.from_numpy(attr_np).to(torch.float32)

        if self.transform is not None:
            attr, target = self.transform(attr, target)
        if self.target_transform is not None:
            target = self.target_transform(target)

        meta: Dict[str, object] = {
            "generated": True,
            "split": self.mode,
            "index": index,
        }
        return attr, target, meta


def build_dataset_from_config(
    *,
    cfg: Dict,
    mode: str = "train",
    transform: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
    target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    seed: Optional[int] = None,
):
    """Factory that returns EEGDataset (real) or GenEEGDataset (synthetic).

    Heuristic: if data.root exists and contains <mode>/Brain, use EEGDataset;
    otherwise use GenEEGDataset with parameters from data.gen[mode].
    """
    data_cfg = cfg.get("data", cfg)
    root = data_cfg.get("root")
    if isinstance(root, str):
        base_dir = os.path.join(root, mode, "Brain")
        if os.path.isdir(base_dir):
            return EEGDataset(
                config=cfg,
                mode=mode,
                transform=transform,
                target_transform=target_transform,
                seed=seed,
            )
    # Fallback to generated dataset
    return GenEEGDataset(
        config=cfg,
        mode=mode,
        transform=transform,
        target_transform=target_transform,
        seed=seed,
    )
        
__all__ = [
    "EEGDataset",
    "GenEEGDataset",
    "build_dataset_from_config",
]
