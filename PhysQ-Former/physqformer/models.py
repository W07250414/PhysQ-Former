from __future__ import annotations

import torch
import torch.nn as nn


def compute_thetae_proxy(t: torch.Tensor, q: torch.Tensor, p_hpa: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    q = torch.clamp(q, min=1e-7, max=0.03)
    r = q / torch.clamp(1.0 - q, min=eps)
    theta = t * torch.pow(1000.0 / torch.clamp(p_hpa, min=1.0), 0.286)
    thetae = theta * torch.exp((2.5e6 * r) / (1004.0 * torch.clamp(t, min=180.0)))
    return thetae


def diff_wrt_logp(x: torch.Tensor, p_hpa: torch.Tensor) -> torch.Tensor:
    """Pressure-coordinate vertical contrast normalized by |Δln p|."""
    logp = torch.log(torch.clamp(p_hpa, min=1.0))
    dx = x[:, 1:] - x[:, :-1]
    dlogp = torch.abs(logp[:, 1:] - logp[:, :-1])
    grad = dx / torch.clamp(dlogp, min=1e-6)
    return torch.cat([grad, grad[:, -1:]], dim=1)


class ResBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x) + self.net(x))


class PhysicsCouplingLayer(nn.Module):
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn, _ = self.mha(x, x, x)
        x = self.norm(x + attn)
        return self.norm2(x + self.ffn(x))


class ImprovedVerticalInstabilityLayer(nn.Module):
    """Pressure-aware vertical thermodynamic representation."""

    def __init__(self, pressure_levels_hpa: list[float], hidden_dim: int = 128, use_thetae_profile: bool = True):
        super().__init__()
        self.use_thetae_profile = use_thetae_profile
        self.register_buffer(
            "pressure_levels",
            torch.tensor(pressure_levels_hpa, dtype=torch.float32).view(1, -1),
        )

        self.per_level_mlp = nn.Sequential(
            nn.Linear(7, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64),
        )

        self.grad_proj = nn.Sequential(
            nn.Linear(4, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, hidden_dim),
        )

        self.conv1d = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1),
        )

        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.grad_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.conv_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, z: torch.Tensor, rh: torch.Tensor, q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        p = self.pressure_levels.to(t.device).expand(batch_size, -1)

        if self.use_thetae_profile:
            thetae = compute_thetae_proxy(t, q, p)
        else:
            thetae = torch.zeros_like(t)

        dt_dlnp = diff_wrt_logp(t, p)
        dq_dlnp = diff_wrt_logp(q, p)
        dz_dlnp = diff_wrt_logp(z, p)
        dthetae_dlnp = diff_wrt_logp(thetae, p)

        level_stack = torch.stack([
            t / 300.0,
            q * 1000.0,
            rh / 100.0,
            z / 1.0e4,
            thetae / 400.0,
            dt_dlnp / 50.0,
            dthetae_dlnp / 100.0,
        ], dim=-1)
        level_embs = self.per_level_mlp(level_stack)

        grad_stack = torch.stack([
            dt_dlnp / 50.0,
            dq_dlnp * 1000.0,
            dz_dlnp / 1.0e4,
            dthetae_dlnp / 100.0,
        ], dim=-1)
        grad_embs = self.grad_proj(grad_stack)

        query = self.query.expand(batch_size, -1, -1)
        grad_feat, _ = self.grad_attn(query, grad_embs, grad_embs)
        grad_feat = grad_feat.squeeze(1)

        conv_input = level_embs.transpose(1, 2)
        conv_feat = self.conv1d(conv_input).transpose(1, 2)
        conv_feat, _ = self.conv_attn(query, conv_feat, conv_feat)
        conv_feat = conv_feat.squeeze(1)

        return self.fusion(torch.cat([grad_feat, conv_feat], dim=1))


class LevelwiseCrossGroupInteraction(nn.Module):
    """Level-wise cross-variable interaction among Z, RH, q, and T."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.per_level_proj = nn.Sequential(
            nn.Linear(4, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, hidden_dim),
        )
        self.level_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, 4, 256, batch_first=True, dropout=0.1),
            num_layers=1,
        )
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pool_attn = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)

    def forward(self, z: torch.Tensor, rh: torch.Tensor, q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        level_stack = torch.stack([
            z / 1.0e4,
            rh / 100.0,
            q * 1000.0,
            t / 300.0,
        ], dim=-1)

        level_feat = self.per_level_proj(level_stack)
        level_feat = self.level_encoder(level_feat)

        query = self.query.expand(batch_size, -1, -1)
        pooled, _ = self.pool_attn(query, level_feat, level_feat)
        return pooled.squeeze(1)


class IntensityOutputMixin:
    def _make_outputs(self, features: torch.Tensor, temperature: float | None = None) -> dict[str, torch.Tensor]:
        if temperature is None:
            temperature = self.cfg["temperature"]

        rain_logit = self.occurrence_head(features)
        heavy_logit = self.heavy_gate_head(features)
        pred_light_log = self.regression_head_light(features)
        pred_heavy_log = self.regression_head_heavy(features)

        rain_prob = torch.sigmoid(rain_logit)
        heavy_prob = torch.sigmoid(heavy_logit / temperature)

        pred_light_raw = torch.expm1(pred_light_log)
        pred_heavy_raw = torch.expm1(pred_heavy_log)

        if self.cfg.get("use_moe", True):
            conditional_intensity_raw = (1.0 - heavy_prob) * pred_light_raw + heavy_prob * pred_heavy_raw
        else:
            heavy_prob = torch.zeros_like(heavy_prob)
            conditional_intensity_raw = pred_light_raw

        conditional_intensity_raw = torch.clamp(conditional_intensity_raw, min=0.0)
        conditional_intensity_log = torch.log1p(conditional_intensity_raw)

        if self.cfg.get("use_rain_prob_for_final", False):
            final_raw = rain_prob * conditional_intensity_raw
        else:
            final_raw = conditional_intensity_raw

        final_raw = torch.clamp(final_raw, min=0.0)
        final_log = torch.log1p(final_raw)

        return {
            "rain_logit": rain_logit,
            "rain_prob": rain_prob,
            "heavy_logit": heavy_logit,
            "heavy_prob": heavy_prob,
            "pred_light_log": pred_light_log,
            "pred_heavy_log": pred_heavy_log,
            "conditional_intensity_log": conditional_intensity_log,
            "final_log": final_log,
            "final_raw": final_raw,
        }


class FlatRainIntensityNet(nn.Module, IntensityOutputMixin):
    """Flattened MLP baseline with the same output dictionary as PhysQ-Former."""

    def __init__(self, scaler_mean, scaler_scale, cfg: dict, hidden_dim: int = 128):
        super().__init__()
        self.cfg = cfg
        self.expected_input_dim = int(cfg["feature_layout"]["expected_input_dim"])
        self.register_buffer("scaler_mean", torch.tensor(scaler_mean, dtype=torch.float32).view(1, -1))
        self.register_buffer("scaler_scale", torch.tensor(scaler_scale, dtype=torch.float32).view(1, -1))

        feat_dim = hidden_dim * 2
        self.backbone = nn.Sequential(
            ResBlock(self.expected_input_dim, hidden_dim),
            ResBlock(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
        )
        self.occurrence_head = nn.Sequential(nn.Linear(feat_dim, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 1))
        self.heavy_gate_head = nn.Sequential(nn.Linear(feat_dim, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 1))
        self.regression_head_light = nn.Sequential(nn.Linear(feat_dim, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1))
        self.regression_head_heavy = nn.Sequential(nn.Linear(feat_dim, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1))

    def forward(self, x: torch.Tensor, temperature: float | None = None) -> dict[str, torch.Tensor]:
        if x.size(1) != self.expected_input_dim:
            raise ValueError(f"Input feature dimension mismatch: got {x.size(1)}, expected {self.expected_input_dim}.")
        features = self.backbone(x)
        return self._make_outputs(features, temperature=temperature)


class PhysicsTransformerAllWeatherNet(nn.Module, IntensityOutputMixin):
    """PhysQ-Former model with structured tokens and intensity-adaptive mapping."""

    def __init__(self, scaler_mean, scaler_scale, cfg: dict, hidden_dim: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = int(hidden_dim or cfg.get("hidden_dim", 128))
        layout = cfg["feature_layout"]
        self.expected_input_dim = int(layout["expected_input_dim"])
        self.slices = layout["slices"]
        group_dims = [
            int(layout["surface_dim"]),
            int(layout["level_dim"]),
            int(layout["level_dim"]),
            int(layout["level_dim"]),
            int(layout["level_dim"]),
        ]

        self.register_buffer("scaler_mean", torch.tensor(scaler_mean, dtype=torch.float32).view(1, -1))
        self.register_buffer("scaler_scale", torch.tensor(scaler_scale, dtype=torch.float32).view(1, -1))

        self.surf = ResBlock(group_dims[0], self.hidden_dim)
        self.projs = nn.ModuleList([
            nn.Sequential(nn.Linear(d, self.hidden_dim), nn.LayerNorm(self.hidden_dim))
            for d in group_dims[1:]
        ])
        self.coupling = PhysicsCouplingLayer(self.hidden_dim)
        self.instability_layer = ImprovedVerticalInstabilityLayer(
            cfg["pressure_levels_hpa"],
            hidden_dim=self.hidden_dim,
            use_thetae_profile=cfg.get("use_thetae_profile", True),
        )
        self.levelwise_cross_group = LevelwiseCrossGroupInteraction(hidden_dim=self.hidden_dim)
        self.tf = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.hidden_dim, 4, 256, batch_first=True, dropout=0.1),
            num_layers=2,
        )

        feat_dim = self.hidden_dim * 7
        self.occurrence_head = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 1)
        )
        self.heavy_gate_head = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 1)
        )
        self.regression_head_light = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1)
        )
        self.regression_head_heavy = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1)
        )

    def recover_physical_x(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scaler_scale + self.scaler_mean

    def _slice(self, x: torch.Tensor, name: str) -> torch.Tensor:
        start, end = self.slices[name]
        return x[:, start:end]

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) != self.expected_input_dim:
            raise ValueError(f"Input feature dimension mismatch: got {x.size(1)}, expected {self.expected_input_dim}.")

        x_phys = self.recover_physical_x(x)

        surf_scaled = self._slice(x, "surface")
        geo_scaled = self._slice(x, "z")
        rh_scaled = self._slice(x, "rh")
        q_scaled = self._slice(x, "q")
        t_scaled = self._slice(x, "t")

        geo_phys = self._slice(x_phys, "z")
        rh_phys = self._slice(x_phys, "rh")
        q_phys = self._slice(x_phys, "q")
        t_phys = self._slice(x_phys, "t")

        surf = self.surf(surf_scaled).unsqueeze(1)
        atmos_embs = [proj(raw) for proj, raw in zip(self.projs, [geo_scaled, rh_scaled, q_scaled, t_scaled])]
        atmos_stack = torch.stack(atmos_embs, dim=1)
        atmos_coupled = self.coupling(atmos_stack)

        if self.cfg.get("use_pressure_aware_vertical", True):
            instability_feat = self.instability_layer(geo_phys, rh_phys, q_phys, t_phys).unsqueeze(1)
        else:
            instability_feat = torch.zeros(surf.size(0), 1, surf.size(-1), device=x.device, dtype=surf.dtype)

        if self.cfg.get("use_levelwise_cross_group", True):
            cross_group_feat = self.levelwise_cross_group(geo_phys, rh_phys, q_phys, t_phys).unsqueeze(1)
        else:
            cross_group_feat = torch.zeros(surf.size(0), 1, surf.size(-1), device=x.device, dtype=surf.dtype)

        seq = torch.cat([surf, atmos_coupled, instability_feat, cross_group_feat], dim=1)
        out = self.tf(seq)
        return out.reshape(out.size(0), -1)

    def forward(self, x: torch.Tensor, temperature: float | None = None) -> dict[str, torch.Tensor]:
        features = self.encode_features(x)
        return self._make_outputs(features, temperature=temperature)


PhysQFormer = PhysicsTransformerAllWeatherNet


def build_model(scaler_mean, scaler_scale, cfg: dict) -> nn.Module:
    model_type = str(cfg.get("model_type", "physqformer")).lower()
    hidden_dim = int(cfg.get("hidden_dim", 128))
    if model_type in {"physqformer", "physics_transformer", "structured"}:
        return PhysicsTransformerAllWeatherNet(scaler_mean, scaler_scale, cfg, hidden_dim=hidden_dim)
    if model_type in {"flat_mlp", "mlp", "flattened"}:
        return FlatRainIntensityNet(scaler_mean, scaler_scale, cfg, hidden_dim=hidden_dim)
    raise ValueError(f"Unsupported model_type={model_type!r}.")
