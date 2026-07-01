from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    if np.allclose(y_true, y_true[0]):
        return float("nan")
    return float(r2_score(y_true, y_pred))


def compute_extended_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    nse = 1.0 - (numerator / denominator) if denominator > 0 else -np.inf

    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        r = np.corrcoef(y_true, y_pred)[0, 1]
    else:
        r = 0.0
    alpha = np.std(y_pred) / np.std(y_true) if np.std(y_true) > 0 else 0.0
    beta = np.mean(y_pred) / np.mean(y_true) if np.mean(y_true) > 0 else 0.0
    kge = 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)

    pbias = 100.0 * np.sum(y_pred - y_true) / np.sum(y_true) if np.sum(y_true) > 0 else 0.0
    return float(nse), float(kge), float(pbias)


def compute_categorical_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> dict[str, float]:
    obs_event = y_true >= threshold
    pred_event = y_pred >= threshold

    hits = np.sum(obs_event & pred_event)
    misses = np.sum(obs_event & (~pred_event))
    false_alarms = np.sum((~obs_event) & pred_event)

    pod = hits / (hits + misses) if (hits + misses) > 0 else 0.0
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else 0.0
    csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else 0.0
    precision = hits / (hits + false_alarms) if (hits + false_alarms) > 0 else 0.0
    recall = pod
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "CSI": float(csi),
        "POD": float(pod),
        "FAR": float(far),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
    }


def predict_loader(model, loader, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    p_list, t_list, rain_prob_list = [], [], []
    with torch.no_grad():
        for x, _, y_raw, _, _ in loader:
            outputs = model(x.to(device))
            pred_raw = outputs["final_raw"].cpu().numpy()
            rain_prob = outputs["rain_prob"].cpu().numpy()
            p_list.append(pred_raw)
            t_list.append(y_raw.numpy())
            rain_prob_list.append(rain_prob)

    y_pred = np.concatenate(p_list).flatten()
    y_true = np.concatenate(t_list).flatten()
    rain_prob = np.concatenate(rain_prob_list).flatten()
    y_pred = np.maximum(y_pred, 0.0)
    return y_true, y_pred, rain_prob


def evaluate_metrics(
    model,
    loader,
    name: str,
    cfg: dict,
    include_zero_row: bool = True,
    report_occurrence_metrics: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    y_true, y_pred, rain_prob = predict_loader(model, loader, cfg["device"])

    if include_zero_row:
        ranges = [
            (0.0, 0.1, "0.0-0.1 (Zero)"),
            (0.1, 2.5, "0.1-2.5 (Light)"),
            (2.5, 5.0, "2.5-5.0 (Mod.)"),
            (5.0, 10.0, "5.0-10.0 (Heavy)"),
            (10.0, 20.0, "10.0-20.0 (Violent)"),
            (20.0, 9999.0, ">20.0 (Extreme)"),
        ]
    else:
        ranges = [
            (0.1, 2.5, "0.1-2.5 (Light)"),
            (2.5, 5.0, "2.5-5.0 (Mod.)"),
            (5.0, 10.0, "5.0-10.0 (Heavy)"),
            (10.0, 20.0, "10.0-20.0 (Violent)"),
            (20.0, 9999.0, ">20.0 (Extreme)"),
        ]

    print(f"\n>>> {name} Report <<<")
    print(f"{'P Range':<20} | {'Samples':<8} | {'MAE':<8} | {'RMSE':<8} | {'MB':<8} | {'R2':<8}")
    print("-" * 82)
    for lower, upper, label in ranges:
        mask = (y_true >= lower) & (y_true < upper)
        if np.sum(mask) > 0:
            yt = y_true[mask]
            yp = y_pred[mask]
            print(
                f"{label:<20} | {len(yt):<8} | {mean_absolute_error(yt, yp):<8.3f} | "
                f"{np.sqrt(mean_squared_error(yt, yp)):<8.3f} | {np.mean(yp - yt):<8.3f} | {safe_r2(yt, yp)!s:<8}"
            )
    print("-" * 82)

    mae_overall = mean_absolute_error(y_true, y_pred)
    rmse_overall = np.sqrt(mean_squared_error(y_true, y_pred))
    mb_overall = np.mean(y_pred - y_true)
    r2_overall = safe_r2(y_true, y_pred)

    print(
        f"{'Overall':<20} | {len(y_true):<8} | {mae_overall:<8.3f} | {rmse_overall:<8.3f} | "
        f"{mb_overall:<8.3f} | {r2_overall!s:<8}"
    )

    mask_heavy = y_true >= cfg["heavy_threshold"]
    if np.sum(mask_heavy) > 0:
        yt_h = y_true[mask_heavy]
        yp_h = y_pred[mask_heavy]
        print(
            f"{'≥5.0mm/hr (Heavy+)':<20} | {len(yt_h):<8} | {mean_absolute_error(yt_h, yp_h):<8.3f} | "
            f"{np.sqrt(mean_squared_error(yt_h, yp_h)):<8.3f} | {np.mean(yp_h - yt_h):<8.3f} | {safe_r2(yt_h, yp_h)!s:<8}"
        )

    print("\n>>> Extended Metrics <<<")
    nse, kge, pbias = compute_extended_metrics(y_true, y_pred)
    print(f"NSE: {nse:.4f} | KGE: {kge:.4f} | PBIAS: {pbias:.2f}%")

    p90 = np.percentile(y_true, 90)
    mask_tail = y_true >= p90
    if np.sum(mask_tail) > 0:
        tail_rmse = np.sqrt(mean_squared_error(y_true[mask_tail], y_pred[mask_tail]))
        print(f"Tail RMSE (>P90={p90:.2f}mm/hr): {tail_rmse:.3f}")

    heavy_cls_metrics = compute_categorical_metrics(y_true, y_pred, cfg["heavy_threshold"])
    print(
        f"Heavy threshold={cfg['heavy_threshold']}mm/hr | CSI: {heavy_cls_metrics['CSI']:.4f} | "
        f"POD: {heavy_cls_metrics['POD']:.4f} | FAR: {heavy_cls_metrics['FAR']:.4f} | F1: {heavy_cls_metrics['F1']:.4f}"
    )

    if report_occurrence_metrics:
        rain_cls_metrics = compute_categorical_metrics(y_true, y_pred, cfg["rain_threshold"])
        print(
            f"Rain threshold={cfg['rain_threshold']}mm/hr | CSI: {rain_cls_metrics['CSI']:.4f} | "
            f"POD: {rain_cls_metrics['POD']:.4f} | FAR: {rain_cls_metrics['FAR']:.4f} | F1: {rain_cls_metrics['F1']:.4f}"
        )
        obs_rain = (y_true >= cfg["rain_threshold"]).astype(np.float32)
        brier = np.mean((rain_prob - obs_rain) ** 2)
        print(f"Occurrence Brier Score: {brier:.5f}")

    return y_true, y_pred


def evaluate_balanced_val_score(model, val_rain_loader, cfg: dict) -> tuple[float, dict[str, float]]:
    y_true_rain, y_pred_rain, _ = predict_loader(model, val_rain_loader, cfg["device"])

    rmse_rain = float(np.sqrt(mean_squared_error(y_true_rain, y_pred_rain)))
    mb_rain = float(np.mean(y_pred_rain - y_true_rain))
    pbias_rain = float(100.0 * np.sum(y_pred_rain - y_true_rain) / np.sum(y_true_rain))

    heavy_mask = y_true_rain >= cfg["heavy_threshold"]
    if np.any(heavy_mask):
        heavy_mb_rain = float(np.mean(y_pred_rain[heavy_mask] - y_true_rain[heavy_mask]))
    else:
        heavy_mb_rain = 0.0

    pos_pbias_penalty = max(pbias_rain, 0.0) / 100.0
    neg_heavy_mb_penalty = max(-heavy_mb_rain, 0.0)

    score = (
        0.65 * rmse_rain
        + 0.15 * abs(mb_rain)
        + 0.10 * pos_pbias_penalty
        + 0.10 * neg_heavy_mb_penalty
    )

    detail = {
        "rmse_rain": rmse_rain,
        "mb_rain": mb_rain,
        "pbias_rain": pbias_rain,
        "heavy_mb_rain": heavy_mb_rain,
        "score": float(score),
    }
    return float(score), detail
