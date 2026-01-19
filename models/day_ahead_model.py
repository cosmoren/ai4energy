import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union, Sequence
from pytorch_lightning import LightningModule


Features = Union[torch.Tensor, Dict[str, torch.Tensor]]


class DayAheadMLP(nn.Module):
    """
    Simple MLP for day-ahead forecasting.

    Expected dataset target is clear-sky index kt.

    Inputs:
      - features (flat): torch.Tensor [B, D]
      - features (structured): dict with
          - endo:   [B, D_endo]
          - nam_cc: [B, H]
          - nam:    [B, T, H]

    Output:
      - kt prediction with shape [B, T, H] ALWAYS
    """

    def __init__(
        self,
        num_targets: int = 2,
        num_horizons: int = 14,
        endo_dim: int = 0,
        hidden_dim: int = 256,
        depth: int = 2,
        dropout: float = 0.1,
        kt_max: float = 1.2,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if endo_dim <= 0:
            raise ValueError("endo_dim must be > 0")

        self.num_targets = num_targets
        self.num_horizons = num_horizons
        self.kt_max = kt_max

        # Separate encoders for each feature group
        enc_dim = hidden_dim
        cc_dim = max(32, hidden_dim // 4)
        nam_dim = max(32, hidden_dim // 4)

        self.endo_enc = nn.Sequential(
            nn.Linear(endo_dim, enc_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cc_enc = nn.Sequential(
            nn.Linear(num_horizons, cc_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Encode NAM per target: [B, T, H] -> [B, T, nam_dim] then pool over targets
        self.nam_enc = nn.Sequential(
            nn.Linear(num_horizons, nam_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        fused_dim = enc_dim + cc_dim + nam_dim
        blocks = []
        for _ in range(depth - 1):
            blocks.extend([nn.Linear(fused_dim, fused_dim), nn.GELU(), nn.Dropout(dropout)])
        self.backbone = nn.Sequential(*blocks) if blocks else nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, num_targets * num_horizons),
        )

    def _flatten_features(self, features: Features) -> torch.Tensor:
        if isinstance(features, dict):
            endo = features["endo"]
            nam_cc = features["nam_cc"]
            nam = features["nam"]
            if nam.dim() == 3:
                nam = nam.flatten(1)  # [B, T*H]
            return torch.cat([endo, nam_cc, nam], dim=1)
        return features

    def forward(self, features: Features) -> torch.Tensor:
        if isinstance(features, dict):
            endo = features["endo"]  # [B, D_endo]
            nam_cc = features["nam_cc"]  # [B, H]
            nam = features["nam"]  # [B, T, H]

            z_endo = self.endo_enc(endo)
            z_cc = self.cc_enc(nam_cc)

            # [B, T, H] -> [B*T, H] -> [B*T, nam_dim] -> [B, T, nam_dim] -> pool over targets
            B = nam.shape[0]
            z_nam = self.nam_enc(nam.reshape(B * self.num_targets, self.num_horizons))
            z_nam = z_nam.view(B, self.num_targets, -1).mean(dim=1)

            z = torch.cat([z_endo, z_cc, z_nam], dim=1)
        else:
            raise ValueError(
                "DayAheadMLP expects structured features dict with keys {'endo','nam_cc','nam'}. "
                "Your dataset is likely returning flat 'features' (possibly from an old cache). "
                "Set feature_return='structured' and/or clear the dataset cache."
            )
        z = self.backbone(z)
        out = self.head(z)
        out = out.view(out.shape[0], self.num_targets, self.num_horizons)
        out = torch.sigmoid(out) * self.kt_max
        return out


class DayAhead(LightningModule):
    """
    Minimal LightningModule wrapper for DayAheadMLP.
    """

    def __init__(
        self,
        num_targets: int = 2,
        num_horizons: int = 14,
        endo_dim: int = 0,
        target_names: Optional[Sequence[str]] = None,
        hidden_dim: int = 256,
        depth: int = 2,
        dropout: float = 0.1,
        kt_max: float = 1.2,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.05,
        beta1: float = 0.9,
        beta2: float = 0.95,
        eps: float = 1e-8,
        loss_beta: float = 0.05,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = DayAheadMLP(
            num_targets=num_targets,
            num_horizons=num_horizons,
            endo_dim=endo_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            kt_max=kt_max,
        )
        self.criterion = nn.SmoothL1Loss(beta=loss_beta, reduction="mean")
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.target_names = list(target_names) if target_names is not None else None

    def forward(self, features: Features) -> torch.Tensor:
        return self.model(features)

    def _features_from_batch(self, batch: Dict[str, torch.Tensor]) -> Features:
        if "features" in batch:
            return batch["features"]
        if "endo" in batch and "nam_cc" in batch and "nam" in batch:
            return {"endo": batch["endo"], "nam_cc": batch["nam_cc"], "nam": batch["nam"]}
        raise KeyError("Batch must contain either 'features' or ('endo','nam_cc','nam').")

    def _ensure_bth(self, y: torch.Tensor) -> torch.Tensor:
        """
        Ensure target tensor is shaped [B, T, H], even if the dataset used squeeze conventions.
        """
        if y.dim() == 3:
            return y
        if y.dim() == 2:
            # Could be [B, H] when T==1 or [B, T] when H==1
            if self.hparams.num_targets == 1 and y.shape[1] == self.hparams.num_horizons:
                return y.unsqueeze(1)
            if self.hparams.num_horizons == 1 and y.shape[1] == self.hparams.num_targets:
                return y.unsqueeze(2)
        if y.dim() == 1:
            # [B] when T==1 and H==1
            return y.view(-1, 1, 1)
        raise ValueError(f"Unexpected target shape: {tuple(y.shape)}")

    def _target_key(self, ti: int) -> str:
        if self.target_names is not None and len(self.target_names) == int(self.hparams.num_targets):
            return str(self.target_names[ti])
        return f"t{ti}"

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        yhat = self(self._features_from_batch(batch))  # [B, T, H]
        y = self._ensure_bth(batch["target"])
        loss = self.criterion(yhat, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        yhat = self(self._features_from_batch(batch))  # [B, T, H]
        y = self._ensure_bth(batch["target"])
        loss = self.criterion(yhat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # Irradiance-level metrics over time (matches training/evaluate_day_ahead.py)
        if all(k in batch for k in ("actual", "nam_irr", "clear_sky", "elevation")):
            clear = self._ensure_bth(batch["clear_sky"])
            y_pred = yhat * clear
            y_true = self._ensure_bth(batch["actual"])
            y_base = self._ensure_bth(batch["nam_irr"])

            elev = batch["elevation"]
            if elev.dim() == 1:
                elev_bh = elev.view(-1, 1)
            else:
                elev_bh = elev
            mask = (elev_bh < 5).unsqueeze(1).expand(-1, self.hparams.num_targets, -1)  # [B,T,H]

            y_true = y_true.masked_fill(mask, torch.nan)
            y_pred = y_pred.masked_fill(mask, torch.nan)
            y_base = y_base.masked_fill(mask, torch.nan)

            err = y_true - y_pred
            err_b = y_true - y_base

            # Per-target summary (ghi/dni, ...)
            rmse_t = torch.sqrt(torch.nanmean(err ** 2, dim=(0, 2)))  # [T]
            mae_t = torch.nanmean(err.abs(), dim=(0, 2))  # [T]
            mbe_t = torch.nanmean(err, dim=(0, 2))  # [T]
            rmse_b_t = torch.sqrt(torch.nanmean(err_b ** 2, dim=(0, 2)))  # [T]
            skill_t = torch.where(
                (rmse_b_t == 0) | torch.isnan(rmse_b_t),
                torch.zeros_like(rmse_t),
                1.0 - rmse_t / rmse_b_t,
            )

            for ti in range(int(rmse_t.shape[0])):
                name = self._target_key(ti)
                self.log(f"val_irr_rmse_{name}", rmse_t[ti], on_step=False, on_epoch=True, logger=True, sync_dist=True)
                self.log(f"val_irr_mae_{name}", mae_t[ti], on_step=False, on_epoch=True, logger=True, sync_dist=True)
                self.log(f"val_irr_mbe_{name}", mbe_t[ti], on_step=False, on_epoch=True, logger=True, sync_dist=True)
                self.log(f"val_skill_vs_nam_{name}", skill_t[ti], on_step=False, on_epoch=True, logger=True, sync_dist=True)

        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        yhat = self(self._features_from_batch(batch))  # [B, T, H]
        y = self._ensure_bth(batch["target"])
        loss = self.criterion(yhat, y)

        # Irradiance-level metrics (matches training/evaluate_day_ahead.py)
        if all(k in batch for k in ("actual", "nam_irr", "clear_sky", "elevation")):
            clear = self._ensure_bth(batch["clear_sky"])
            y_pred = yhat * clear
            y_true = self._ensure_bth(batch["actual"])
            y_base = self._ensure_bth(batch["nam_irr"])

            elev = batch["elevation"]
            if elev.dim() == 1:
                elev_bh = elev.view(-1, 1)
            else:
                elev_bh = elev
            mask = (elev_bh < 5).unsqueeze(1).expand(-1, self.hparams.num_targets, -1)  # [B,T,H]

            y_true = y_true.masked_fill(mask, torch.nan)
            y_pred = y_pred.masked_fill(mask, torch.nan)
            y_base = y_base.masked_fill(mask, torch.nan)

            err = y_true - y_pred
            err_b = y_true - y_base

            # Per-target summary (ghi/dni, ...)
            rmse_t = torch.sqrt(torch.nanmean(err ** 2, dim=(0, 2)))  # [T]
            mae_t = torch.nanmean(err.abs(), dim=(0, 2))  # [T]
            mbe_t = torch.nanmean(err, dim=(0, 2))  # [T]
            rmse_b_t = torch.sqrt(torch.nanmean(err_b ** 2, dim=(0, 2)))  # [T]
            skill_t = torch.where(
                (rmse_b_t == 0) | torch.isnan(rmse_b_t),
                torch.zeros_like(rmse_t),
                1.0 - rmse_t / rmse_b_t,
            )

            for ti in range(int(rmse_t.shape[0])):
                name = self._target_key(ti)
                self.log(f"test_irr_rmse_{name}", rmse_t[ti], on_step=False, on_epoch=True, logger=True, sync_dist=True)
                self.log(f"test_irr_mae_{name}", mae_t[ti], on_step=False, on_epoch=True, logger=True, sync_dist=True)
                self.log(f"test_irr_mbe_{name}", mbe_t[ti], on_step=False, on_epoch=True, logger=True, sync_dist=True)
                self.log(f"test_skill_vs_nam_{name}", skill_t[ti], on_step=False, on_epoch=True, logger=True, sync_dist=True)

        return loss

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0):
        return self(self._features_from_batch(batch))  # [B, T, H]

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay,
            eps=self.eps,
        )

