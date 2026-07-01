from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

from .evaluation import safe_r2


def plot_line_fit_random(y_true, y_pred, title, ax, num_samples: int = 300):
    total = len(y_true)
    if total > num_samples:
        indices = np.random.choice(total, num_samples, replace=False)
        indices = np.sort(indices)
        yt = y_true[indices]
        yp = y_pred[indices]
        x_axis = np.arange(num_samples)
    else:
        yt = y_true
        yp = y_pred
        x_axis = np.arange(total)

    rmse = np.sqrt(mean_squared_error(yt, yp))
    r2 = safe_r2(yt, yp)

    ax.plot(x_axis, yt, color="black", label="True Value", linewidth=1.5, alpha=0.8)
    ax.plot(x_axis, yp, color="red", label="Predicted", linestyle="--", linewidth=1.5, alpha=0.9)
    ax.fill_between(x_axis, yt, yp, color="gray", alpha=0.1)
    ax.set_title(f"{title}\n(Random {len(yt)} Samples | RMSE: {rmse:.2f} | R2: {r2})")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Precipitation (mm/hr)")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)


def plot_reliability_curve(y_true, y_pred, ax, n_bins: int = 10):
    quantiles = np.linspace(0, 1, n_bins + 1)
    pred_bins = np.quantile(y_pred, quantiles)

    bin_means_pred = []
    bin_means_true = []

    for i in range(n_bins):
        mask = (y_pred >= pred_bins[i]) & (y_pred < pred_bins[i + 1])
        if i == n_bins - 1:
            mask = (y_pred >= pred_bins[i]) & (y_pred <= pred_bins[i + 1])
        if np.sum(mask) > 0:
            bin_means_pred.append(np.mean(y_pred[mask]))
            bin_means_true.append(np.mean(y_true[mask]))

    bin_means_pred = np.array(bin_means_pred)
    bin_means_true = np.array(bin_means_true)

    max_val = max(bin_means_pred.max(), bin_means_true.max()) if len(bin_means_pred) > 0 else 1.0
    ax.plot([0, max_val], [0, max_val], "k--", label="Perfect Calibration", alpha=0.5)
    ax.scatter(bin_means_pred, bin_means_true, s=100, alpha=0.7, c="blue", edgecolors="black")
    ax.plot(bin_means_pred, bin_means_true, "b-", alpha=0.5)
    ax.set_xlabel("Mean Predicted Value (mm/hr)")
    ax.set_ylabel("Mean Observed Value (mm/hr)")
    ax.set_title("Reliability Calibration Curve\n(Binned by Prediction Quantiles)")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)


def plot_loglog_scatter(y_true, y_pred, ax, n_samples: int = 5000, n_bins: int = 20):
    if len(y_true) > n_samples:
        idx = np.random.choice(len(y_true), n_samples, replace=False)
        yt = y_true[idx]
        yp = y_pred[idx]
    else:
        yt = y_true
        yp = y_pred

    yt_log = np.log1p(yt)
    yp_log = np.log1p(yp)
    ax.scatter(yt_log, yp_log, alpha=0.3, s=10, c="blue", edgecolors="none")

    bins = np.linspace(yt_log.min(), yt_log.max(), n_bins + 1)
    centers = []
    means = []
    for i in range(n_bins):
        mask = (yt_log >= bins[i]) & (yt_log < bins[i + 1])
        if i == n_bins - 1:
            mask = (yt_log >= bins[i]) & (yt_log <= bins[i + 1])
        if np.sum(mask) > 0:
            centers.append((bins[i] + bins[i + 1]) / 2)
            means.append(np.mean(yp_log[mask]))

    ax.plot(centers, means, "r-", linewidth=2, label="Binned Regression")
    max_val = max(yt_log.max(), yp_log.max())
    ax.plot([0, max_val], [0, max_val], "k--", label="Perfect Prediction", alpha=0.5)
    ax.set_xlabel("log(1 + True) (mm/hr)")
    ax.set_ylabel("log(1 + Predicted) (mm/hr)")
    ax.set_title("Log-Log Scatter Plot with Binned Regression")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)


def save_experiment_figures(y_tr_t, y_tr_p, y_te_t, y_te_p, cfg: dict):
    print("\n>>> Generating diagnostic figures...")
    fig1, axes1 = plt.subplots(1, 2, figsize=(18, 6))
    plot_line_fit_random(y_tr_t, y_tr_p, f"Training Fit ({cfg['ablation_name']})", axes1[0], num_samples=300)
    plot_line_fit_random(y_te_t, y_te_p, f"Test Fit ({cfg['ablation_name']})", axes1[1], num_samples=300)
    plt.tight_layout()
    plt.savefig(cfg["line_fit_png"], dpi=300)
    print(f">>> Saved: {cfg['line_fit_png']}")
    plt.close(fig1)

    fig2, axes2 = plt.subplots(1, 2, figsize=(18, 6))
    plot_reliability_curve(y_te_t, y_te_p, axes2[0], n_bins=10)
    plot_loglog_scatter(y_te_t, y_te_p, axes2[1], n_samples=5000, n_bins=20)
    plt.tight_layout()
    plt.savefig(cfg["diag_png"], dpi=300)
    print(f">>> Saved: {cfg['diag_png']}")
    plt.close(fig2)
