from __future__ import annotations

import argparse
import os

from torch.utils.data import DataLoader

from IO import build_dataset_from_config
from preprocess import collate_eeg_batch_channel
from train import Trainer, build_noam_from_config, LossComputer, MetricsComputer
from model import build_model_from_config


def load_yaml(path: str) -> dict:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise ImportError("PyYAML is required to load YAML configs. Install with `pip install pyyaml`.") from e
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_loaders(cfg: dict, seed: int | None) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}
    # data_loader can live under train.data_loader (preferred) or at top-level for backward-compat
    if isinstance(train_cfg, dict) and "data_loader" in train_cfg:
        dl_cfg = train_cfg.get("data_loader", {}) or {}
    else:
        dl_cfg = cfg.get("data_loader", {}) if isinstance(cfg, dict) else {}

    batch_size = int(dl_cfg.get("batch_size", 32))
    num_workers = int(dl_cfg.get("num_workers", 4))
    pin_memory = bool(dl_cfg.get("pin_memory", True))

    train_ds = build_dataset_from_config(cfg=cfg, mode="train", seed=seed)
    val_ds = build_dataset_from_config(cfg=cfg, mode="val", seed=seed)
    test_ds = build_dataset_from_config(cfg=cfg, mode="test", seed=seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=pin_memory,
        collate_fn=collate_eeg_batch_channel,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_eeg_batch_channel,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_eeg_batch_channel,
    )
    return train_loader, val_loader, test_loader

def main() -> None:
    parser = argparse.ArgumentParser(description="Train ArtifactRemovalTransformer")
    parser.add_argument("--config", default="./config.yaml", help="Path to YAML config")
    parser.add_argument("--save-dir", default=None, help="Override save dir (read from config if not set)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs from config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset generation/shuffling")
    args = parser.parse_args()

    # Load config once
    cfg = load_yaml(args.config)

    # Build loaders externally (flexible)
    train_loader, val_loader, test_loader = build_loaders(cfg=cfg, seed=args.seed)

    # Build model and optimizer from config outside the Trainer
    model = build_model_from_config(cfg=cfg)
    opt = build_noam_from_config(cfg=cfg, model=model)

    # Resolve save_dir from config, allow CLI override
    cfg_train = cfg.get("train", {}) if isinstance(cfg, dict) else {}
    save_dir = args.save_dir or cfg_train.get("save_dir")

    # Build loss and metrics from config
    loss_comp = LossComputer(cfg)
    metrics_comp = MetricsComputer(cfg)

    trainer = Trainer(
        cfg=cfg,
        save_dir=save_dir,
        model=model,
        opt=opt,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        loss_comp=loss_comp,
        metrics_comp=metrics_comp,
    )

    trainer.fit(epochs=args.epochs)


if __name__ == "__main__":
    main()
