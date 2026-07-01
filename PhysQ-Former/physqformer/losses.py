from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedAsymmetricLoss(nn.Module):
    """Log-space regression loss with intensity-dependent weights and heavy-rain pinball term."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("bin_penalties", torch.tensor([1.0, 1.0, 1.1, 1.3, 1.7, 2.2], dtype=torch.float32))
        self.register_buffer("class_weights", torch.tensor([1.2, 1.0, 1.2, 1.5, 2.0, 2.6], dtype=torch.float32))

    def forward(self, pred_log: torch.Tensor, target_log: torch.Tensor) -> torch.Tensor:
        diff = pred_log - target_log
        target_raw = torch.expm1(target_log)
        pred_raw = torch.expm1(pred_log)

        penalties = torch.ones_like(diff)
        class_weight_map = torch.ones_like(diff)
        bins = self.cfg["bins"]

        for i in range(len(bins) - 1):
            mask = (target_raw >= bins[i]) & (target_raw < bins[i + 1])
            penalties[mask] = self.bin_penalties[i]
            class_weight_map[mask] = self.class_weights[i]

        weights = torch.where(diff < 0, penalties, 1.0)
        loss_mse = (diff ** 2) * weights * class_weight_map

        tau = 0.9
        mask_heavy = target_raw >= self.cfg["heavy_threshold"]
        if torch.sum(mask_heavy) > 0:
            diff_raw_heavy = target_raw[mask_heavy] - pred_raw[mask_heavy]
            pinball_loss_heavy = torch.where(
                diff_raw_heavy > 0,
                tau * diff_raw_heavy,
                (tau - 1.0) * diff_raw_heavy,
            )
            pinball_mean = torch.mean(pinball_loss_heavy)
        else:
            pinball_mean = torch.tensor(0.0, device=pred_log.device)

        relative_loss = ((pred_raw - target_raw) ** 2) / (target_raw + 1.0)
        return torch.mean(loss_mse) + 0.10 * pinball_mean + 0.05 * torch.mean(relative_loss)


class AllWeatherOccurrenceIntensityLoss(nn.Module):
    """Training objective for auxiliary occurrence and conditional rainfall intensity."""

    def __init__(self, cfg: dict, heavy_pos_weight: float = 2.0):
        super().__init__()
        self.cfg = cfg
        self.reg_loss = EnhancedAsymmetricLoss(cfg)
        self.occ_bce = nn.BCEWithLogitsLoss()
        self.register_buffer("heavy_pos_weight", torch.tensor([heavy_pos_weight], dtype=torch.float32))
        self.heavy_bce = nn.BCEWithLogitsLoss(pos_weight=self.heavy_pos_weight)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        target_log: torch.Tensor,
        target_raw: torch.Tensor,
        occ_target: torch.Tensor,
        heavy_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        rain_logit = outputs["rain_logit"]
        heavy_logit = outputs["heavy_logit"]
        pred_light_log = outputs["pred_light_log"]
        pred_heavy_log = outputs["pred_heavy_log"]
        conditional_intensity_log = outputs["conditional_intensity_log"]
        final_log = outputs["final_log"]
        final_raw = outputs["final_raw"]

        loss_occ = self.occ_bce(rain_logit, occ_target)

        mask_rain = occ_target.squeeze(1) > 0.5
        mask_heavy = heavy_target.squeeze(1) > 0.5
        mask_light_rain = mask_rain & (~mask_heavy)

        if self.cfg.get("use_moe", True):
            if torch.sum(mask_light_rain) > 0:
                loss_light = self.reg_loss(pred_light_log[mask_light_rain], target_log[mask_light_rain])
            else:
                loss_light = torch.tensor(0.0, device=target_log.device)

            if torch.sum(mask_heavy) > 0:
                loss_heavy = self.reg_loss(pred_heavy_log[mask_heavy], target_log[mask_heavy])
            else:
                loss_heavy = torch.tensor(0.0, device=target_log.device)

            if torch.sum(mask_rain) > 0:
                loss_cond_mix = self.reg_loss(conditional_intensity_log[mask_rain], target_log[mask_rain])
                loss_heavy_gate = self.heavy_bce(heavy_logit[mask_rain], heavy_target[mask_rain])
            else:
                loss_cond_mix = torch.tensor(0.0, device=target_log.device)
                loss_heavy_gate = torch.tensor(0.0, device=target_log.device)
        else:
            if torch.sum(mask_rain) > 0:
                loss_light = self.reg_loss(pred_light_log[mask_rain], target_log[mask_rain])
            else:
                loss_light = torch.tensor(0.0, device=target_log.device)
            loss_heavy = torch.tensor(0.0, device=target_log.device)
            loss_cond_mix = torch.tensor(0.0, device=target_log.device)
            loss_heavy_gate = torch.tensor(0.0, device=target_log.device)

        loss_final_allweather = self.reg_loss(final_log, target_log)

        mask_dry = occ_target.squeeze(1) < 0.5
        if torch.sum(mask_dry) > 0:
            dry_margin = 0.03
            dry_excess = F.relu(final_raw[mask_dry] - dry_margin)
            dry_penalty = torch.mean(dry_excess ** 2)
        else:
            dry_penalty = torch.tensor(0.0, device=target_log.device)

        mask_light_mod = (target_raw.squeeze(1) >= self.cfg["rain_threshold"]) & (
            target_raw.squeeze(1) < self.cfg["heavy_threshold"]
        )
        if torch.sum(mask_light_mod) > 0:
            true_lm = target_raw[mask_light_mod]
            pred_lm = final_raw[mask_light_mod]
            rel_over = F.relu(pred_lm - true_lm) / (true_lm + 1.0)
            loss_pos_bias = torch.mean(rel_over ** 2)
        else:
            loss_pos_bias = torch.tensor(0.0, device=target_log.device)

        total = (
            self.cfg["lambda_occ"] * loss_occ
            + loss_light
            + self.cfg["beta_heavy_reg"] * loss_heavy
            + self.cfg["lambda_heavy_gate"] * loss_heavy_gate
            + self.cfg["lambda_cond_mix"] * loss_cond_mix
            + self.cfg["lambda_final_allweather"] * loss_final_allweather
            + self.cfg["lambda_dry_penalty"] * dry_penalty
            + self.cfg["lambda_pos_bias"] * loss_pos_bias
        )

        return total, {
            "loss_total": float(total.detach().item()),
            "loss_occ": float(loss_occ.detach().item()),
            "loss_light": float(loss_light.detach().item()),
            "loss_heavy": float(loss_heavy.detach().item()),
            "loss_heavy_gate": float(loss_heavy_gate.detach().item()),
            "loss_cond_mix": float(loss_cond_mix.detach().item()),
            "loss_final_allweather": float(loss_final_allweather.detach().item()),
            "loss_dry_penalty": float(dry_penalty.detach().item()),
            "loss_pos_bias": float(loss_pos_bias.detach().item()),
        }
