"""
PyTorch Lightning interface for intra_hour_model.

This module provides a Lightning wrapper for the intra_hour_model,
simplifying training, validation, and inference workflows.
"""

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from typing import Optional, Dict, Any
import torch.nn.functional as F

from .intra_hour_model import intra_hour_model


class IntraHourLightningModule(LightningModule):
    """
    PyTorch Lightning module wrapper for intra_hour_model.
    
    This wrapper provides:
    - Automatic training/validation step logic
    - Optimizer and scheduler configuration
    - Logging integration
    - Checkpointing support
    """
    
    def __init__(
        self,
        # Model hyperparameters
        image_size: int = 448,
        num_frames: int = 30,
        video_embed_dim: int = 1024,
        output_channels: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        # Training hyperparameters
        learning_rate: float = 3e-4,
        weight_decay: float = 0.05,
        beta1: float = 0.9,
        beta2: float = 0.95,
        eps: float = 1e-8,
        loss_beta: float = 0.05,
        # Scheduler (optional)
        use_scheduler: bool = False,
        scheduler_type: str = "cosine",
        warmup_epochs: int = 0,
        **kwargs
    ):
        """
        Initialize the Lightning module.
        
        Args:
            image_size: Size of input images (default: 448)
            num_frames: Number of frames (default: 30)
            video_embed_dim: Embedding dimension from VideoTransformer (default: 1024)
            output_channels: Number of output channels (default: 2)
            hidden_dim: Hidden dimension for fusion layers (default: 256)
            dropout: Dropout rate (default: 0.1)
            learning_rate: Learning rate for optimizer (default: 3e-4)
            weight_decay: Weight decay for optimizer (default: 0.05)
            beta1: Beta1 for AdamW optimizer (default: 0.9)
            beta2: Beta2 for AdamW optimizer (default: 0.95)
            eps: Epsilon for AdamW optimizer (default: 1e-8)
            loss_beta: Beta parameter for SmoothL1Loss (default: 0.05)
            use_scheduler: Whether to use learning rate scheduler (default: False)
            scheduler_type: Type of scheduler ("cosine", "step", "plateau") (default: "cosine")
            warmup_epochs: Number of warmup epochs (default: 0)
            **kwargs: Additional arguments passed to intra_hour_model
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        # Initialize the model
        self.model = intra_hour_model(
            image_size=image_size,
            num_frames=num_frames,
            video_embed_dim=video_embed_dim,
            output_channels=output_channels,
            hidden_dim=hidden_dim,
            dropout=dropout,
            **kwargs
        )
        
        # Loss function
        self.criterion = nn.SmoothL1Loss(beta=loss_beta, reduction="mean")
        
        # Training hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.use_scheduler = use_scheduler
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        
    def forward(
        self, 
        images: torch.Tensor, 
        irradiance: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            images: Input tensor of shape [B, T, C, H, W]
            irradiance: Optional irradiance features of shape [B, ...]
        
        Returns:
            Output tensor of shape [B, output_channels, T]
        """
        return self.model(images, irradiance)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Dictionary containing 'images', 'irradiance', and 'target'
            batch_idx: Batch index
        
        Returns:
            Loss tensor
        """
        images = batch['images']
        irradiance = batch['irradiance']
        target = batch['target']
        
        # Forward pass
        outputs = self(images, irradiance)
        
        # Compute loss
        loss = self.criterion(outputs, target)
        
        # Logging
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Validation step.
        
        Args:
            batch: Dictionary containing 'images', 'irradiance', and 'target'
            batch_idx: Batch index
        
        Returns:
            Loss tensor
        """
        images = batch['images']
        irradiance = batch['irradiance']
        target = batch['target']
        
        # Forward pass
        outputs = self(images, irradiance)
        
        # Compute loss
        loss = self.criterion(outputs, target)
        
        # Compute additional metrics (MAE, RMSE)
        with torch.no_grad():
            mae = F.l1_loss(outputs, target)
            mse = F.mse_loss(outputs, target)
            rmse = torch.sqrt(mse)
        
        # Logging
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/mae', mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/rmse', rmse, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Test step.
        
        Args:
            batch: Dictionary containing 'images', 'irradiance', and 'target'
            batch_idx: Batch index
        
        Returns:
            Loss tensor
        """
        images = batch['images']
        irradiance = batch['irradiance']
        target = batch['target']
        
        # Forward pass
        outputs = self(images, irradiance)
        
        # Compute loss
        loss = self.criterion(outputs, target)
        
        # Compute additional metrics
        with torch.no_grad():
            mae = F.l1_loss(outputs, target)
            mse = F.mse_loss(outputs, target)
            rmse = torch.sqrt(mse)
        
        # Logging
        self.log('test/loss', loss, on_step=False, on_epoch=True, logger=True)
        self.log('test/mae', mae, on_step=False, on_epoch=True, logger=True)
        self.log('test/rmse', rmse, on_step=False, on_epoch=True, logger=True)
        
        return loss
    
    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        
        Returns:
            Optimizer and optionally scheduler configuration
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay,
            eps=self.eps
        )
        
        if not self.use_scheduler:
            return optimizer
        
        # Configure scheduler
        if self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs if self.trainer.max_epochs else 100,
                eta_min=1e-6
            )
        elif self.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=10,
                gamma=0.1
            )
        elif self.scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")
        
        # Configure scheduler return value
        if self.scheduler_type == "plateau":
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss"
                }
            }
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler
            }
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Prediction step for inference.
        
        Args:
            batch: Dictionary containing 'images' and optionally 'irradiance'
            batch_idx: Batch index
        
        Returns:
            Model predictions
        """
        images = batch['images']
        irradiance = batch.get('irradiance', None)
        
        with torch.no_grad():
            predictions = self(images, irradiance)
        
        return predictions
