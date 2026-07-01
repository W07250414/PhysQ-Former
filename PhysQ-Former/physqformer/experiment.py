from __future__ import annotations

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader

from .config import ABLATION_EXPERIMENTS, CONFIG, apply_experiment_config
from .data import RainfallDataset, build_weighted_sampler, load_and_preprocess
from .evaluation import evaluate_balanced_val_score, evaluate_metrics, safe_r2
from .losses import AllWeatherOccurrenceIntensityLoss
from .models import build_model
from .plotting import save_experiment_figures
from .utils import format_seconds, set_seed


def _make_loaders(data: dict, cfg: dict) -> dict[str, DataLoader]:
    train_all_ds = RainfallDataset(data["X_train_all"], data["y_train_all_log"], data["y_train_all_raw"], cfg)
    val_all_ds = RainfallDataset(data["X_val_all"], data["y_val_all_log"], data["y_val_all_raw"], cfg)
    train_rain_ds = RainfallDataset(data["X_train_rain"], data["y_train_rain_log"], data["y_train_rain_raw"], cfg)
    val_rain_ds = RainfallDataset(data["X_val_rain"], data["y_val_rain_log"], data["y_val_rain_raw"], cfg)
    test_rain_ds = RainfallDataset(data["X_test_rain"], data["y_test_rain_log"], data["y_test_rain_raw"], cfg)

    train_sampler = build_weighted_sampler(data["y_train_all_raw"], cfg)

    return {
        "train": DataLoader(train_all_ds, batch_size=cfg["batch_size"], sampler=train_sampler),
        "train_eval": DataLoader(train_all_ds, batch_size=cfg["batch_size"], shuffle=False),
        "val_all": DataLoader(val_all_ds, batch_size=cfg["batch_size"], shuffle=False),
        "train_rain": DataLoader(train_rain_ds, batch_size=cfg["batch_size"], shuffle=False),
        "val_rain": DataLoader(val_rain_ds, batch_size=cfg["batch_size"], shuffle=False),
        "test_rain": DataLoader(test_rain_ds, batch_size=cfg["batch_size"], shuffle=False),
    }


def _heavy_pos_weight(y_train_all: np.ndarray, cfg: dict) -> float:
    rain_mask = y_train_all >= cfg["rain_threshold"]
    heavy_mask = y_train_all >= cfg["heavy_threshold"]
    heavy_count = max(int(np.sum(heavy_mask)), 1)
    light_or_mod_count = max(int(np.sum(rain_mask & (~heavy_mask))), 1)
    return float(min(light_or_mod_count / heavy_count, 10.0))


def run_single_experiment(exp_cfg: dict | None = None) -> dict:
    cfg = apply_experiment_config(CONFIG, exp_cfg or {})

    set_seed(cfg["random_seed"])
    print(f">>> Random seed set to {cfg['random_seed']}")
    print(f">>> Device: {cfg['device']}")
    print(
        f">>> Strategy: {cfg['ablation_name']} | model_type={cfg['model_type']} | "
        f"pressure_vertical={cfg['use_pressure_aware_vertical']} | "
        f"cross_group={cfg['use_levelwise_cross_group']} | MoE={cfg['use_moe']}"
    )

    data = load_and_preprocess(cfg)
    loaders = _make_loaders(data, cfg)

    model = build_model(data["scaler_mean"], data["scaler_scale"], cfg).to(cfg["device"])
    criterion = AllWeatherOccurrenceIntensityLoss(
        cfg=cfg,
        heavy_pos_weight=_heavy_pos_weight(data["y_train_all_raw"], cfg),
    ).to(cfg["device"])

    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg["lr"] * 2,
        steps_per_epoch=max(1, len(loaders["train"])),
        epochs=cfg["epochs"],
    )

    best_val_score = np.inf
    best_epoch = 0
    train_start_time = time.time()

    print(f">>> Start training: {cfg['ablation_name']} <<<")
    for ep in range(cfg["epochs"]):
        epoch_start_time = time.time()
        model.train()
        epoch_losses, epoch_occ, epoch_dry, epoch_pos_bias = [], [], [], []

        for x, y_log, y_raw, occ, heavy in loaders["train"]:
            x = x.to(cfg["device"])
            y_log = y_log.to(cfg["device"])
            y_raw = y_raw.to(cfg["device"])
            occ = occ.to(cfg["device"])
            heavy = heavy.to(cfg["device"])

            optimizer.zero_grad()
            outputs = model(x)
            loss, loss_dict = criterion(outputs, y_log, y_raw, occ, heavy)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())
            epoch_occ.append(loss_dict["loss_occ"])
            epoch_dry.append(loss_dict["loss_dry_penalty"])
            epoch_pos_bias.append(loss_dict["loss_pos_bias"])

        balanced_val_score, val_detail = evaluate_balanced_val_score(model, loaders["val_rain"], cfg)

        if cfg["device"] == "cuda":
            torch.cuda.synchronize()

        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - train_start_time
        avg_epoch_time = total_time / (ep + 1)
        eta_time = avg_epoch_time * (cfg["epochs"] - ep - 1)
        current_lr = optimizer.param_groups[0]["lr"]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(
            f"[{timestamp}] Epoch {ep + 1:03d}/{cfg['epochs']:03d} | "
            f"train_loss={np.mean(epoch_losses):.5f} | "
            f"occ={np.mean(epoch_occ):.5f} | dry={np.mean(epoch_dry):.5f} | "
            f"pos_bias={np.mean(epoch_pos_bias):.5f} | "
            f"val_score={balanced_val_score:.5f} | val_rmse={val_detail['rmse_rain']:.4f} | "
            f"val_mb={val_detail['mb_rain']:.4f} | val_pbias={val_detail['pbias_rain']:.2f}% | "
            f"val_heavy_mb={val_detail['heavy_mb_rain']:.4f} | lr={current_lr:.2e} | "
            f"epoch_time={format_seconds(epoch_time)} | elapsed={format_seconds(total_time)} | eta={format_seconds(eta_time)}",
            flush=True,
        )

        if cfg["save_best"] and balanced_val_score < best_val_score:
            best_val_score = balanced_val_score
            best_epoch = ep + 1
            torch.save(
                {
                    "epoch": ep + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "balanced_val_score": balanced_val_score,
                    "val_detail": val_detail,
                    "config": cfg,
                },
                cfg["checkpoint_path"],
            )
            print(
                f">>> Best model saved at epoch {ep + 1} | path={cfg['checkpoint_path']} | "
                f"score={balanced_val_score:.4f} | val_rmse={val_detail['rmse_rain']:.4f}"
            )

    print(
        f"\n>>> Training complete. Best epoch: {best_epoch} | "
        f"Best balanced_score={best_val_score:.4f} | Total time={format_seconds(time.time() - train_start_time)}"
    )

    if cfg["save_best"] and os.path.exists(cfg["checkpoint_path"]):
        checkpoint = torch.load(cfg["checkpoint_path"], map_location=cfg["device"], weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f">>> Loaded best model: epoch={checkpoint['epoch']} | "
            f"balanced_score={checkpoint.get('balanced_val_score', np.nan):.4f}"
        )

    print("\n" + "=" * 90)
    print(f">>> Evaluation A: Training | {cfg['ablation_name']}")
    print("=" * 90)
    y_tr_t, y_tr_p = evaluate_metrics(
        model, loaders["train_eval"], "Training", cfg, include_zero_row=True, report_occurrence_metrics=False
    )

    print("\n" + "=" * 90)
    print(f">>> Evaluation B: Validation Rain-Only | {cfg['ablation_name']}")
    print("=" * 90)
    evaluate_metrics(model, loaders["val_rain"], "Validation Rain-Only", cfg, include_zero_row=False)

    print("\n" + "=" * 90)
    print(f">>> Evaluation C: Test Rain-Only | {cfg['ablation_name']}")
    print("=" * 90)
    y_te_t, y_te_p = evaluate_metrics(model, loaders["test_rain"], "Test Rain-Only", cfg, include_zero_row=False)

    save_experiment_figures(y_tr_t, y_tr_p, y_te_t, y_te_p, cfg)

    heavy_mask = y_te_t >= cfg["heavy_threshold"]
    result = {
        "ablation_name": cfg["ablation_name"],
        "model_type": cfg["model_type"],
        "use_pressure_aware_vertical": cfg["use_pressure_aware_vertical"],
        "use_levelwise_cross_group": cfg["use_levelwise_cross_group"],
        "use_moe": cfg["use_moe"],
        "test_overall_mae": float(mean_absolute_error(y_te_t, y_te_p)),
        "test_overall_rmse": float(np.sqrt(mean_squared_error(y_te_t, y_te_p))),
        "test_overall_mb": float(np.mean(y_te_p - y_te_t)),
        "test_overall_r2": float(safe_r2(y_te_t, y_te_p)),
        "test_heavy_rmse": float(np.sqrt(mean_squared_error(y_te_t[heavy_mask], y_te_p[heavy_mask]))) if np.any(heavy_mask) else np.nan,
        "test_heavy_mb": float(np.mean(y_te_p[heavy_mask] - y_te_t[heavy_mask])) if np.any(heavy_mask) else np.nan,
        "best_epoch": int(best_epoch),
        "best_balanced_score": float(best_val_score),
    }
    return result


def run_ablation_suite() -> pd.DataFrame:
    results = []
    for exp_cfg in ABLATION_EXPERIMENTS:
        print("\n" + "#" * 100)
        print(f">>> Running ablation: {exp_cfg['ablation_name']}")
        print("#" * 100)
        results.append(run_single_experiment(exp_cfg))

    df = pd.DataFrame(results)
    summary_csv = apply_experiment_config(CONFIG, {})["summary_csv"]
    df.to_csv(summary_csv, index=False)
    print(f"\n>>> Ablation summary saved to: {summary_csv}")
    print(df)
    return df
