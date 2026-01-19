
import torch
import torch.nn as nn
from typing import Optional, Dict
from pytorch_lightning import LightningModule
import torch.nn.functional as F

from .intra_hour_model import intra_hour_model

class IntraHour(LightningModule):
    def __init__(
        self,
        image_size: int = 224,
        num_frames: int = 30,
        video_embed_dim: int = 1024,
        output_channels: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.05,
        beta1: float = 0.9,
        beta2: float = 0.95,
        eps: float = 1e-8,
        loss_beta: float = 0.05,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = intra_hour_model(
            image_size=image_size,
            num_frames=num_frames,
            video_embed_dim=video_embed_dim,
            output_channels=output_channels,
            hidden_dim=hidden_dim,
            dropout=dropout,
            **kwargs,
        )

        self.criterion = nn.SmoothL1Loss(beta=loss_beta, reduction="mean")
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def forward(self, images: torch.Tensor, irradiance: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.model(images, irradiance)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        yhat = self(batch["images"], batch["irradiance"])
        loss = self.criterion(yhat, batch["target"])
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        yhat = self(batch["images"], batch["irradiance"])
        y = batch["target"]
        loss = self.criterion(yhat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/mae", F.l1_loss(yhat, y), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/rmse", torch.sqrt(F.mse_loss(yhat, y)), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/mbe", torch.mean(yhat - y), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        yhat = self(batch["images"], batch["irradiance"])
        y = batch["target"]
        loss = self.criterion(yhat, y)
        self.log("test/loss", loss, on_step=False, on_epoch=True, logger=True)
        self.log("test/mae", F.l1_loss(yhat, y), on_step=False, on_epoch=True, logger=True)
        self.log("test/rmse", torch.sqrt(F.mse_loss(yhat, y)), on_step=False, on_epoch=True, logger=True)
        self.log("test/mbe", torch.mean(yhat - y), on_step=False, on_epoch=True, logger=True)
        return loss

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0):
        return self(batch["images"], batch["irradiance"])

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay,
            eps=self.eps,
        )
