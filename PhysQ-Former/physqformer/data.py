from __future__ import annotations

import gc
import os
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, WeightedRandomSampler


def expected_input_dim(cfg: dict) -> int:
    return int(cfg["feature_layout"]["expected_input_dim"])


def collect_existing_file_paths(file_ids: Iterable[int], data_dir: str, tag: str) -> list[str]:
    file_paths: list[str] = []
    missing_ids: list[int] = []
    for i in file_ids:
        f_path = os.path.join(data_dir, f"train_data_{i}.pkl")
        if os.path.exists(f_path):
            file_paths.append(f_path)
        else:
            missing_ids.append(i)

    if not file_paths:
        raise ValueError(f"{tag} has no available data files. Please check data_dir and file_ids.")

    if missing_ids:
        print(f">>> [Warning] {tag} missing file ids: {missing_ids}")

    print(f">>> {tag}: found {len(file_paths)} file(s)")
    return file_paths


def _validate_feature_dim(X: np.ndarray, fp: str, cfg: dict) -> None:
    expected = expected_input_dim(cfg)
    actual = int(X.shape[1])
    if actual != expected:
        raise ValueError(
            f"Feature dimension mismatch in {fp}: got {actual}, expected {expected}. "
            "Expected order: 11 surface variables + 19 Z + 19 RH + 19 q + 19 T."
        )


def read_feature_target_from_paths(file_paths: list[str], cfg: dict) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Read pkl files whose first column is precipitation and remaining columns are predictors."""
    X_by_file, y_by_file = [], []
    for fp in file_paths:
        df_chunk = pd.read_pickle(fp)
        X = df_chunk.iloc[:, 1:].values.astype(np.float32)
        y = df_chunk.iloc[:, 0].values.astype(np.float32)
        _validate_feature_dim(X, fp, cfg)
        print(f">>> Loaded {os.path.basename(fp)} | X shape={X.shape} | y shape={y.shape}")
        X_by_file.append(X)
        y_by_file.append(y)
        del df_chunk
        gc.collect()
    return X_by_file, y_by_file


def fill_and_scale(X_subset: np.ndarray, col_mean: np.ndarray, scaler: StandardScaler, fit: bool = False) -> np.ndarray:
    X_filled = X_subset.copy()
    inds = np.where(np.isnan(X_filled))
    X_filled[inds] = np.take(col_mean, inds[1])
    if fit:
        return scaler.fit_transform(X_filled).astype(np.float32)
    return scaler.transform(X_filled).astype(np.float32)


def split_rain_data(
    X_rain_by_file: list[np.ndarray],
    y_rain_by_file: list[np.ndarray],
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split_mode = str(cfg.get("split_mode", "random")).lower()
    val_ratio = float(cfg.get("val_ratio", 0.1))
    test_ratio = float(cfg.get("test_ratio", 0.1))

    if split_mode == "by_file":
        rng = np.random.RandomState(cfg["random_seed"])
        order = np.arange(len(X_rain_by_file))
        rng.shuffle(order)

        n_total = len(order)
        n_test = max(1, int(round(n_total * test_ratio)))
        n_val = max(1, int(round(n_total * val_ratio)))
        n_train = max(1, n_total - n_val - n_test)

        idx_train = order[:n_train]
        idx_val = order[n_train:n_train + n_val]
        idx_test = order[n_train + n_val:]

        X_train = np.concatenate([X_rain_by_file[k] for k in idx_train], axis=0)
        y_train = np.concatenate([y_rain_by_file[k] for k in idx_train], axis=0)
        X_val = np.concatenate([X_rain_by_file[k] for k in idx_val], axis=0)
        y_val = np.concatenate([y_rain_by_file[k] for k in idx_val], axis=0)
        X_test = np.concatenate([X_rain_by_file[k] for k in idx_test], axis=0)
        y_test = np.concatenate([y_rain_by_file[k] for k in idx_test], axis=0)
    elif split_mode == "random":
        X_rain = np.concatenate(X_rain_by_file, axis=0)
        y_rain = np.concatenate(y_rain_by_file, axis=0)

        rain_mask = y_rain >= cfg["rain_threshold"]
        X_rain = X_rain[rain_mask]
        y_rain = y_rain[rain_mask]

        X_train, X_temp, y_train, y_temp = train_test_split(
            X_rain,
            y_rain,
            test_size=(val_ratio + test_ratio),
            random_state=cfg["random_seed"],
        )

        test_frac_in_temp = test_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=test_frac_in_temp,
            random_state=cfg["random_seed"],
        )
    else:
        raise ValueError(f"Unsupported split_mode={split_mode!r}; use 'random' or 'by_file'.")

    train_mask = y_train >= cfg["rain_threshold"]
    val_mask = y_val >= cfg["rain_threshold"]
    test_mask = y_test >= cfg["rain_threshold"]

    return (
        X_train[train_mask], y_train[train_mask],
        X_val[val_mask], y_val[val_mask],
        X_test[test_mask], y_test[test_mask],
    )


def split_zero_data(X_zero: np.ndarray, y_zero: np.ndarray, cfg: dict):
    zero_mask = (y_zero >= 0.0) & (y_zero < cfg["rain_threshold"])
    X_zero = X_zero[zero_mask]
    y_zero = y_zero[zero_mask]

    if len(y_zero) == 0:
        return None, None, None, None

    zero_val_ratio = float(cfg.get("zero_val_ratio", 0.0))
    if zero_val_ratio <= 0.0:
        return X_zero, y_zero, None, None

    X_zero_train, X_zero_val, y_zero_train, y_zero_val = train_test_split(
        X_zero,
        y_zero,
        test_size=zero_val_ratio,
        random_state=cfg["random_seed"],
    )
    return X_zero_train, y_zero_train, X_zero_val, y_zero_val


def load_and_preprocess(cfg: dict) -> dict:
    """Load CAPD-style pkl files and return scaled train/val/test arrays."""
    print(
        ">>> [Step 1] Loading data | train_data_1-12 for rainy split; "
        "train_data_13 for synthetic dry augmentation if available."
    )

    rain_paths = collect_existing_file_paths(cfg["rain_file_ids"], cfg["data_dir"], "Rain files")
    X_rain_by_file, y_rain_by_file = read_feature_target_from_paths(rain_paths, cfg)

    X_train_rain, y_train_rain, X_val_rain, y_val_rain, X_test_rain, y_test_rain = split_rain_data(
        X_rain_by_file, y_rain_by_file, cfg
    )
    del X_rain_by_file, y_rain_by_file
    gc.collect()

    zero_paths = collect_existing_file_paths(cfg["zero_rain_file_ids"], cfg["data_dir"], "Synthetic dry files")
    X_zero_by_file, y_zero_by_file = read_feature_target_from_paths(zero_paths, cfg)
    X_zero = np.concatenate(X_zero_by_file, axis=0)
    y_zero = np.concatenate(y_zero_by_file, axis=0)
    del X_zero_by_file, y_zero_by_file
    gc.collect()

    X_zero_train, y_zero_train, X_zero_val, y_zero_val = split_zero_data(X_zero, y_zero, cfg)
    del X_zero, y_zero
    gc.collect()

    if X_zero_train is not None and len(X_zero_train) > 0:
        X_train_all = np.concatenate([X_train_rain, X_zero_train], axis=0)
        y_train_all = np.concatenate([y_train_rain, y_zero_train], axis=0)
    else:
        X_train_all = X_train_rain.copy()
        y_train_all = y_train_rain.copy()

    if X_zero_val is not None and len(X_zero_val) > 0:
        X_val_all = np.concatenate([X_val_rain, X_zero_val], axis=0)
        y_val_all = np.concatenate([y_val_rain, y_zero_val], axis=0)
    else:
        X_val_all = X_val_rain.copy()
        y_val_all = y_val_rain.copy()

    for arr in [X_train_all, X_val_all, X_test_rain, X_train_rain, X_val_rain]:
        arr[arr < -9000] = np.nan

    # Use the synthetic-augmented training set for exactly reproducing the current training script.
    # To avoid synthetic dry samples in standardization, replace X_train_all with X_train_rain here.
    col_mean = np.nanmean(X_train_all, axis=0)
    col_mean[np.isnan(col_mean)] = 0.0
    scaler_x = StandardScaler()

    X_train_all_scaled = fill_and_scale(X_train_all, col_mean, scaler_x, fit=True)
    X_val_all_scaled = fill_and_scale(X_val_all, col_mean, scaler_x, fit=False)
    X_train_rain_scaled = fill_and_scale(X_train_rain, col_mean, scaler_x, fit=False)
    X_val_rain_scaled = fill_and_scale(X_val_rain, col_mean, scaler_x, fit=False)
    X_test_rain_scaled = fill_and_scale(X_test_rain, col_mean, scaler_x, fit=False)

    y_train_all_log = np.log1p(y_train_all).astype(np.float32)
    y_val_all_log = np.log1p(y_val_all).astype(np.float32)
    y_train_rain_log = np.log1p(y_train_rain).astype(np.float32)
    y_val_rain_log = np.log1p(y_val_rain).astype(np.float32)
    y_test_rain_log = np.log1p(y_test_rain).astype(np.float32)

    train_zero_count = int(np.sum((y_train_all >= 0.0) & (y_train_all < cfg["rain_threshold"])))
    train_rain_count = int(np.sum(y_train_all >= cfg["rain_threshold"]))
    heavy_count = int(np.sum(y_train_all >= cfg["heavy_threshold"]))

    print(
        f"Train(synthetic-augmented): total={len(y_train_all)}, "
        f"synthetic_zero={train_zero_count}, real_rain={train_rain_count}, heavy={heavy_count}"
    )
    print(f"Val(real rain-only): total={len(y_val_rain)}")
    print(f"Test(real rain-only): total={len(y_test_rain)}")

    return {
        "X_train_all": X_train_all_scaled,
        "y_train_all_log": y_train_all_log,
        "y_train_all_raw": y_train_all.astype(np.float32),
        "X_val_all": X_val_all_scaled,
        "y_val_all_log": y_val_all_log,
        "y_val_all_raw": y_val_all.astype(np.float32),
        "X_train_rain": X_train_rain_scaled,
        "y_train_rain_log": y_train_rain_log,
        "y_train_rain_raw": y_train_rain.astype(np.float32),
        "X_val_rain": X_val_rain_scaled,
        "y_val_rain_log": y_val_rain_log,
        "y_val_rain_raw": y_val_rain.astype(np.float32),
        "X_test_rain": X_test_rain_scaled,
        "y_test_rain_log": y_test_rain_log,
        "y_test_rain_raw": y_test_rain.astype(np.float32),
        "scaler_mean": scaler_x.mean_.astype(np.float32),
        "scaler_scale": scaler_x.scale_.astype(np.float32),
        "col_mean": col_mean.astype(np.float32),
    }


class RainfallDataset(Dataset):
    def __init__(self, X: np.ndarray, y_log: np.ndarray, y_raw: np.ndarray, cfg: dict):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_log = torch.tensor(y_log, dtype=torch.float32).unsqueeze(1)
        self.y_raw = torch.tensor(y_raw, dtype=torch.float32).unsqueeze(1)
        self.occ = (self.y_raw >= cfg["rain_threshold"]).float()
        self.heavy = (self.y_raw >= cfg["heavy_threshold"]).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y_log[idx], self.y_raw[idx], self.occ[idx], self.heavy[idx]


def build_weighted_sampler(y_raw: np.ndarray, cfg: dict) -> WeightedRandomSampler:
    weights = np.ones(len(y_raw), dtype=np.float32)

    zero_mask = y_raw < cfg["rain_threshold"]
    light_mask = (y_raw >= cfg["rain_threshold"]) & (y_raw < 2.5)
    mod_mask = (y_raw >= 2.5) & (y_raw < cfg["heavy_threshold"])
    heavy_mask = (y_raw >= cfg["heavy_threshold"]) & (y_raw < cfg["violent_threshold"])
    violent_mask = (y_raw >= cfg["violent_threshold"]) & (y_raw < cfg["extreme_threshold"])
    extreme_mask = y_raw >= cfg["extreme_threshold"]

    weights[zero_mask] = cfg["sampler_zero_weight"]
    weights[light_mask] = cfg["sampler_light_weight"]
    weights[mod_mask] = cfg["sampler_mod_weight"]
    weights[heavy_mask] = cfg["sampler_heavy_weight"]
    weights[violent_mask] = cfg["sampler_violent_weight"]
    weights[extreme_mask] = cfg["sampler_extreme_weight"]

    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )
