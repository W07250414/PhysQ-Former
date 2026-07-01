from __future__ import annotations

import copy
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The atmospheric input follows the paper formulation:
# 11 single-level predictors + 4 upper-air profiles * 19 pressure levels = 87 features.
FEATURE_LAYOUT = {
    "surface_dim": 11,
    "level_dim": 19,
    "expected_input_dim": 87,
    "groups": ["surface", "z", "rh", "q", "t"],
    "slices": {
        "surface": (0, 11),
        "z": (11, 30),
        "rh": (30, 49),
        "q": (49, 68),
        "t": (68, 87),
    },
}

CONFIG = {
    # ---------------- Data ----------------
    "data_dir": str(PROJECT_ROOT / "data"),
    "rain_file_ids": list(range(1, 13)),
    "zero_rain_file_ids": [13],
    "rain_threshold": 0.1,
    "heavy_threshold": 5.0,
    "split_mode": "random",  # choices: random, by_file
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "zero_val_ratio": 0.0,
    "feature_layout": FEATURE_LAYOUT,

    # ---------------- Training ----------------
    "batch_size": 1024,
    "epochs": 80,
    "lr": 5e-4,
    "weight_decay": 1e-3,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "random_seed": 42,
    "save_best": True,
    "checkpoint_dir": str(PROJECT_ROOT / "checkpoints"),
    "output_dir": str(PROJECT_ROOT / "outputs"),
    "checkpoint_path": str(PROJECT_ROOT / "checkpoints" / "best_model_full_physqformer_87dim.pth"),

    # ---------------- Synthetic dry / sampler ----------------
    "sampler_zero_weight": 2.0,
    "sampler_light_weight": 1.0,
    "sampler_mod_weight": 1.3,
    "sampler_heavy_weight": 2.5,
    "sampler_violent_weight": 4.0,
    "sampler_extreme_weight": 8.0,
    "violent_threshold": 10.0,
    "extreme_threshold": 20.0,

    # ---------------- Loss weights ----------------
    "lambda_occ": 0.10,
    "lambda_heavy_gate": 0.15,
    "lambda_cond_mix": 0.10,
    "lambda_final_allweather": 0.45,
    "lambda_dry_penalty": 0.15,
    "beta_heavy_reg": 1.20,
    "lambda_pos_bias": 0.35,
    "temperature": 1.25,

    # ---------------- Inference control ----------------
    # The main rainy-sample task uses conditional intensity directly.
    # The occurrence branch is used as auxiliary training regularization.
    "use_rain_prob_for_final": False,
    "use_thetae_profile": True,

    # ---------------- Ablation switches ----------------
    "ablation_name": "A4_full_physqformer",
    "model_type": "physqformer",
    "hidden_dim": 128,
    "use_pressure_aware_vertical": True,
    "use_levelwise_cross_group": True,
    "use_moe": True,
    "run_ablation_suite": False,

    # ---------------- Pressure levels ----------------
    "pressure_levels_hpa": [
        1000.0, 975.0, 950.0, 925.0, 900.0, 850.0, 800.0, 750.0, 700.0,
        650.0, 600.0, 550.0, 500.0, 450.0, 400.0, 350.0, 300.0, 200.0, 100.0,
    ],

    # ---------------- Loss bins ----------------
    "bins": [0.0, 0.1, 2.5, 5.0, 10.0, 20.0, 9999.0],

    # ---------------- Figures ----------------
    "line_fit_png": str(PROJECT_ROOT / "outputs" / "Line_Fit_A4_full_physqformer.png"),
    "diag_png": str(PROJECT_ROOT / "outputs" / "Diagnostic_Plots_A4_full_physqformer.png"),
    "summary_csv": str(PROJECT_ROOT / "outputs" / "experiment_summary.csv"),
}

# Progressive variants aligned with the manuscript description.
ABLATION_EXPERIMENTS = [
    {
        "ablation_name": "A0_flattened_mlp",
        "model_type": "flat_mlp",
        "use_pressure_aware_vertical": False,
        "use_levelwise_cross_group": False,
        "use_moe": False,
    },
    {
        "ablation_name": "A1_structured_cross",
        "model_type": "physqformer",
        "use_pressure_aware_vertical": False,
        "use_levelwise_cross_group": True,
        "use_moe": False,
    },
    {
        "ablation_name": "A2_pressure_vertical",
        "model_type": "physqformer",
        "use_pressure_aware_vertical": True,
        "use_levelwise_cross_group": True,
        "use_moe": False,
    },
    {
        "ablation_name": "A3_pressure_vertical_moe",
        "model_type": "physqformer",
        "use_pressure_aware_vertical": True,
        "use_levelwise_cross_group": True,
        "use_moe": True,
    },
    {
        "ablation_name": "A4_full_physqformer",
        "model_type": "physqformer",
        "use_pressure_aware_vertical": True,
        "use_levelwise_cross_group": True,
        "use_moe": True,
    },
]


def apply_experiment_config(base_config: dict, exp_cfg: dict | None = None) -> dict:
    """Return an isolated config for one experiment and generate run-specific paths."""
    cfg = copy.deepcopy(base_config)
    if exp_cfg:
        cfg.update(exp_cfg)

    tag = cfg["ablation_name"]
    checkpoint_dir = Path(cfg["checkpoint_dir"])
    output_dir = Path(cfg["output_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg["checkpoint_path"] = str(checkpoint_dir / f"best_model_{tag}.pth")
    cfg["line_fit_png"] = str(output_dir / f"Line_Fit_{tag}.png")
    cfg["diag_png"] = str(output_dir / f"Diagnostic_Plots_{tag}.png")
    cfg["summary_csv"] = str(output_dir / "experiment_summary.csv")
    return cfg
