from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader

from common import (
    AestheticDataset,
    AestheticLoss,
    backbone_parameter_groups,
    build_model,
    count_parameters,
    device_info,
    load_config,
    resolve,
    score_agreement_label,
    score_to_unit,
    set_backbone_trainable,
    set_seed,
    unit_to_score,
)


def split_frame(frame: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = frame["identity"] if "identity" in frame else frame["image_id"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=cfg["test_fraction"], random_state=cfg["seed"])
    train_val_i, test_i = next(splitter.split(frame, groups=groups))
    train_val, test = frame.iloc[train_val_i], frame.iloc[test_i]
    relative_val = cfg["val_fraction"] / (1 - cfg["test_fraction"])
    groups = train_val["identity"] if "identity" in train_val else train_val["image_id"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=relative_val, random_state=cfg["seed"])
    train_i, val_i = next(splitter.split(train_val, groups=groups))
    return train_val.iloc[train_i].copy(), train_val.iloc[val_i].copy(), test.copy()


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_mae = 0.0
    use_amp = device.type == "cuda"
    for x, target_unit, _ in loader:
        x = x.to(device, non_blocking=True)
        target_unit = target_unit.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            raw = model(x)
            pred_unit = torch.sigmoid(raw)
            loss = criterion(pred_unit, target_unit)

        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                loss.backward()
                optimizer.step()

        batch_size = x.size(0)
        total_loss += float(loss.item()) * batch_size
        pred_score = unit_to_score(pred_unit.detach())
        target_score = unit_to_score(target_unit.detach())
        total_mae += torch.abs(pred_score - target_score).sum().item()
    return total_loss / len(loader.dataset), total_mae / len(loader.dataset)


def make_loaders(train_frame, val_frame, cfg, device):
    common_kwargs = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], pin_memory=device.type == "cuda")
    train_loader = DataLoader(AestheticDataset(train_frame, cfg["image_size"], train=True), shuffle=True, **common_kwargs)
    val_loader = DataLoader(AestheticDataset(val_frame, cfg["image_size"], train=False), shuffle=False, **common_kwargs)
    return train_loader, val_loader


def train_single_model(name: str, seed: int, train_frame: pd.DataFrame, val_frame: pd.DataFrame, cfg: dict, device: torch.device) -> dict:
    set_seed(seed)
    model = build_model(name).to(device)
    freeze_epochs = int(cfg.get("freeze_epochs", 3))
    backbone_lr_scale = float(cfg.get("backbone_lr_scale", 0.1))
    set_backbone_trainable(model, False)
    backbone_params, head_params = backbone_parameter_groups(model)
    optimizer = torch.optim.Adam(
        [
            {"params": head_params, "lr": cfg["learning_rate"]},
            {"params": backbone_params, "lr": cfg["learning_rate"] * backbone_lr_scale},
        ],
        weight_decay=cfg["weight_decay"],
    )
    sched_type = str(cfg.get("scheduler", "cosine")).lower()
    if sched_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(cfg["epochs"]), eta_min=float(cfg.get("min_lr", 1e-6))
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=int(cfg.get("lr_patience", 2)), min_lr=1e-6
        )
    pearson_weight = float(cfg.get("pearson_loss_weight", 0.15))
    criterion = AestheticLoss(beta=0.05, pearson_weight=pearson_weight)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    train_loader, val_loader = make_loaders(train_frame, val_frame, cfg, device)
    print(f"Run seed: {seed}; trainable parameters: {count_parameters(model):,}")
    best_val_mae = float("inf")
    stale_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, cfg["epochs"] + 1):
        if epoch == freeze_epochs + 1:
            set_backbone_trainable(model, True)
            print(f"Unfroze backbone at epoch {epoch}; trainable parameters: {count_parameters(model):,}")
        train_loss, train_mae = run_epoch(model, train_loader, criterion, device, optimizer=optimizer, scaler=scaler)
        val_loss, val_mae = run_epoch(model, val_loader, criterion, device, optimizer=None, scaler=None)
        if sched_type == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_mae)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_mae": train_mae,
                "val_mae": val_mae,
                "lr_head": optimizer.param_groups[0]["lr"],
                "lr_backbone": optimizer.param_groups[1]["lr"],
            }
        )
        print(f"{epoch:03d}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, val MAE {val_mae:.4f}")
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            stale_epochs = 0
            save_checkpoint(model, cfg, name, seed, best_val_mae)
        else:
            stale_epochs += 1
            if stale_epochs >= cfg["patience"]:
                print("Early stopping.")
                break

    return {"history": pd.DataFrame(history), "best_val_mae": best_val_mae}


def save_checkpoint(model, cfg, name: str, seed: int, best_val_mae: float) -> Path:
    models_dir = resolve(cfg, "models")
    models_dir.mkdir(parents=True, exist_ok=True)
    if int(cfg.get("ensemble_size", 1)) > 1:
        checkpoint_path = models_dir / f"{name}_seed{seed}_best.pth"
    else:
        checkpoint_path = models_dir / f"{name}_best.pth"
    torch.save(
        {
            "model_name": name,
            "seed": seed,
            "model_state": model.state_dict(),
            "config": cfg,
            "val_mae": best_val_mae,
        },
        checkpoint_path,
    )
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--ensemble-size", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    name = args.model or cfg["model_name"]
    ensemble_size = int(args.ensemble_size or cfg.get("ensemble_size", 1))

    set_seed(cfg["seed"])
    device = device_info()
    manifest = resolve(cfg, cfg["data"]["processed_manifest"])
    if not manifest.exists():
        raise FileNotFoundError("Missing manifest. Run: python src/prepare_data.py")

    frame = pd.read_csv(manifest)
    train_frame, val_frame, test_frame = split_frame(frame, cfg)
    print(f"Dataset: {len(frame)} | train={len(train_frame)} val={len(val_frame)} test={len(test_frame)}")
    root = manifest.parent
    train_frame.to_csv(root / "train.csv", index=False)
    val_frame.to_csv(root / "val.csv", index=False)
    test_frame.to_csv(root / "test.csv", index=False)

    seeds = [cfg["seed"] + i for i in range(ensemble_size)]
    run_results = []
    for seed in seeds:
        result = train_single_model(name, seed, train_frame, val_frame, cfg, device)
        run_results.append((seed, result))
        history = result["history"]
        result_dir = resolve(cfg, "results")
        result_dir.mkdir(parents=True, exist_ok=True)
        history.to_csv(result_dir / (f"{name}_seed{seed}_training_metrics.csv" if ensemble_size > 1 else f"{name}_training_metrics.csv"), index=False)
        plt.figure(figsize=(7, 4))
        plt.plot(history.epoch, history.train_loss, label="train")
        plt.plot(history.epoch, history.val_loss, label="validation")
        plt.xlabel("Epoch")
        plt.ylabel("Huber loss")
        plt.title(f"{name} training loss (seed {seed})")
        plt.legend()
        plt.tight_layout()
        plot_name = f"{name}_seed{seed}_loss.png" if ensemble_size > 1 else f"{name}_loss.png"
        plt.savefig(result_dir / plot_name, dpi=160)
        plt.close()

    summary = pd.DataFrame(
        {
            "seed": [seed for seed, _ in run_results],
            "best_val_mae": [result["best_val_mae"] for _, result in run_results],
        }
    )
    result_dir = resolve(cfg, "results")
    summary.to_csv(result_dir / f"{name}_ensemble_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Best validation MAE: {summary.best_val_mae.min():.4f}")


if __name__ == "__main__":
    main()
