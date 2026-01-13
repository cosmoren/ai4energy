"""
PVInsight: A generalist architecture for solar irradiance forecasting.

This module provides a configurable architecture with separate components for:
- Video encoder: Processes sequence of image frames
- Irradiance encoder: Processes time-series irradiance features
- Fusion module: Combines multiple feature representations
- Prediction head: Generates forecasts

Designed for easy ablation studies and configuration.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple
from pytorch_lightning import LightningModule


class VideoEncoder(nn.Module):
    """
    Video encoder module for learning video feature representations.
    
    Input: [B, T, C, H, W] - batch of video sequences
    Output: [B, T_out, h_dim] - where T_out can be 1 if temporal pooling is applied
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int],  # (C, H, W)
        num_frames: int,
        h_dim: int,
        temporal_pooling: str = "mean",  # "mean", "last", "none"
        backbone: str = "simvp",  # "simvp", "timesformer", "conv3d", "resnet3d", etc.
        **backbone_kwargs
    ):
        """
        Initialize video encoder.
        
        Args:
            input_shape: Input image shape (C, H, W)
            num_frames: Number of input frames (T)
            h_dim: Output hidden dimension
            temporal_pooling: Type of temporal pooling ("mean", "last", "none")
            backbone: Backbone architecture name
            **backbone_kwargs: Additional arguments for backbone
        """
        super().__init__()
        self.input_shape = input_shape
        self.num_frames = num_frames
        self.h_dim = h_dim
        self.temporal_pooling = temporal_pooling
        self.backbone = backbone
        
        # Initialize backbone
        if backbone == "simvp":
            # Import SimVP here to avoid circular imports
            import sys
            from pathlib import Path
            models_dir = Path(__file__).parent
            if str(models_dir) not in sys.path:
                sys.path.insert(0, str(models_dir))
            from .simvp.models import SimVP_Model
            
            self.encoder = SimVP_Model([num_frames, input_shape[0], input_shape[1], input_shape[2]])
            # Extract features from encoder output
            # SimVP encoder outputs [B*T, C, H, W], we need to process this
            # For now, use encoder + spatial pooling
            self.spatial_pool = nn.AdaptiveAvgPool2d(1)
            self.proj = nn.Linear(16, h_dim)  # Assuming SimVP output is 16 channels
        elif backbone == "timesformer":
            # TimeSformer from Hugging Face transformers
            try:
                from transformers import TimesformerModel, TimesformerConfig
            except ImportError:
                raise ImportError(
                    "TimeSformer requires transformers library. Install with: pip install transformers"
                )
            
            # Check if pretrained model is specified
            pretrained_model = backbone_kwargs.get("pretrained_model", None)
            
            if pretrained_model:
                # Load pretrained TimeSformer model
                # Available models: "facebook/timesformer-base-finetuned-k400", 
                #                   "facebook/timesformer-base-finetuned-k600"
                self.encoder = TimesformerModel.from_pretrained(pretrained_model)
                # Get actual hidden size from pretrained model
                pretrained_h_dim = self.encoder.config.hidden_size
                # Project to desired h_dim if different
                if pretrained_h_dim != h_dim:
                    self.proj = nn.Linear(pretrained_h_dim, h_dim)
                else:
                    self.proj = None
            else:
                # Initialize TimeSformer from scratch
                # hidden_size must be divisible by num_attention_heads
                # For h_dim=256, we use 8 heads (256/8=32) or 16 heads (256/16=16)
                num_heads = backbone_kwargs.get("num_attention_heads", 8)
                if h_dim % num_heads != 0:
                    # Adjust num_heads to be compatible with h_dim
                    # Find the largest divisor <= 12
                    num_heads = max([d for d in [8, 16, 32] if h_dim % d == 0], default=8)
                
                config = TimesformerConfig(
                    image_size=input_shape[1],  # H
                    patch_size=16,  # Default patch size
                    num_channels=input_shape[0],  # C
                    num_frames=num_frames,
                    hidden_size=h_dim,  # Use h_dim as hidden_size
                    num_hidden_layers=12,  # Default number of layers
                    num_attention_heads=num_heads,  # Must divide hidden_size
                    intermediate_size=h_dim * 4,  # Standard transformer scaling
                    attention_type="divided_space_time",  # TimeSformer's key feature
                    **{k: v for k, v in backbone_kwargs.items() if k not in ["pretrained_model", "num_attention_heads"]}
                )
                
                self.encoder = TimesformerModel(config)
                self.proj = None  # Already in h_dim
            
            # TimeSformer outputs [B, sequence_length, hidden_size]
            # We'll use CLS token or mean pooling over sequence
            self.use_cls_token = backbone_kwargs.get("use_cls_token", False)
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, T, C, H, W]
        
        Returns:
            Output tensor [B, T_out, h_dim] where T_out depends on temporal_pooling
        """
        B, T, C, H, W = x.shape
        
        # Reshape for backbone: [B, T, C, H, W] -> [B*T, C, H, W]
        x_reshaped = x.view(B * T, C, H, W)
        
        # Encode through backbone
        if self.backbone == "simvp":
            embed, skip = self.encoder.enc(x_reshaped)  # [B*T, C_, H_, W_]
            # Reshape back: [B*T, C_, H_, W_] -> [B, T, C_, H_, W_]
            embed = embed.view(B, T, *embed.shape[1:])
            
            # Spatial pooling: [B, T, C_, H_, W_] -> [B, T, C_, 1, 1]
            embed_pooled = self.spatial_pool(embed.view(B * T, *embed.shape[2:]))
            embed_pooled = embed_pooled.view(B, T, -1)  # [B, T, C_]
            
            # Project to h_dim: [B, T, C_] -> [B, T, h_dim]
            features = self.proj(embed_pooled)
        
        elif self.backbone == "timesformer":
            # TimeSformer expects input in format [B, T, C, H, W] (batch, frames, channels, height, width)
            # Input x is already in format [B, T, C, H, W], so use it directly
            # TimeSformer internally handles patching and positional encoding
            
            # TimeSformer processes the video and outputs last_hidden_state
            # Shape: [B, sequence_length, hidden_size]
            # sequence_length = num_patches * num_frames + 1 (CLS token) or num_patches * num_frames
            outputs = self.encoder(pixel_values=x)  # x is already [B, T, C, H, W]
            hidden_states = outputs.last_hidden_state  # [B, seq_len, hidden_size]
            
            # Extract features: use CLS token or mean pool over sequence
            if self.use_cls_token and hidden_states.shape[1] > 0:
                # Use CLS token (first token): [B, 1, hidden_size]
                features = hidden_states[:, 0:1, :]
            else:
                # Mean pool over sequence: [B, seq_len, hidden_size] -> [B, 1, hidden_size]
                features = hidden_states.mean(dim=1, keepdim=True)
            
            # Project to h_dim if needed (when using pretrained with different hidden_size)
            if self.proj is not None:
                features = self.proj(features)  # [B, 1, h_dim]
        
        else:
            raise NotImplementedError(f"Backbone {self.backbone} not implemented")
        
        # Apply temporal pooling if specified
        if self.temporal_pooling == "mean":
            features = features.mean(dim=1, keepdim=True)  # [B, 1, h_dim]
        elif self.temporal_pooling == "last":
            features = features[:, -1:, :]  # [B, 1, h_dim]
        elif self.temporal_pooling == "none":
            pass  # [B, T, h_dim]
        else:
            raise ValueError(f"Unknown temporal_pooling: {self.temporal_pooling}")
        
        return features


class IrradianceEncoder(nn.Module):
    """
    Irradiance multichannel encoder for processing time-series features.
    
    Input: [B, T, f] - batch, time, features
    Output: [B, h_dim] - encoded features
    """
    
    def __init__(
        self,
        input_features: int,  # f - number of input features
        h_dim: int,
        num_timesteps: Optional[int] = None,  # T - if None, handles variable length
        encoder_type: str = "tcn",  # "tcn", "lstm", "transformer", "mlp"
        num_layers: int = 2,
        **encoder_kwargs
    ):
        """
        Initialize irradiance encoder.
        
        Args:
            input_features: Number of input features (f)
            h_dim: Output hidden dimension
            num_timesteps: Number of timesteps (T), optional
            encoder_type: Type of encoder ("tcn", "lstm", "transformer", "mlp")
            num_layers: Number of layers
            **encoder_kwargs: Additional arguments for encoder
        """
        super().__init__()
        self.input_features = input_features
        self.h_dim = h_dim
        self.num_timesteps = num_timesteps
        self.encoder_type = encoder_type
        
        if encoder_type == "tcn":
            # Temporal Convolutional Network
            layers = []
            in_dim = input_features
            for i in range(num_layers):
                layers.append(nn.Conv1d(in_dim, h_dim, kernel_size=3, padding=1))
                layers.append(nn.GELU())
                in_dim = h_dim
            self.encoder = nn.Sequential(*layers)
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif encoder_type == "lstm":
            # LSTM encoder
            self.encoder = nn.LSTM(
                input_features,
                h_dim,
                num_layers=num_layers,
                batch_first=True,
                **encoder_kwargs
            )
            self.pool = None  # Use last output
        elif encoder_type == "transformer":
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_features,
                nhead=encoder_kwargs.get("nhead", 4),
                dim_feedforward=encoder_kwargs.get("dim_feedforward", h_dim),
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.proj = nn.Linear(input_features, h_dim)
            self.pool = None
        elif encoder_type == "mlp":
            # Simple MLP (assumes temporal pooling is done before)
            self.encoder = nn.Sequential(
                nn.Linear(input_features, h_dim),
                nn.GELU(),
                nn.Linear(h_dim, h_dim)
            )
            self.pool = None
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, T, f]
        
        Returns:
            Output tensor [B, h_dim]
        """
        if self.encoder_type == "tcn":
            # [B, T, f] -> [B, f, T]
            x = x.transpose(1, 2)
            # Encode: [B, f, T] -> [B, h_dim, T]
            x = self.encoder(x)
            # Pool: [B, h_dim, T] -> [B, h_dim, 1] -> [B, h_dim]
            x = self.pool(x).squeeze(-1)
        elif self.encoder_type == "lstm":
            # [B, T, f] -> LSTM -> [B, T, h_dim]
            x, (h_n, c_n) = self.encoder(x)
            # Use last hidden state: [B, h_dim]
            x = h_n[-1]
        elif self.encoder_type == "transformer":
            # [B, T, f] -> Transformer -> [B, T, f]
            x = self.encoder(x)
            # Mean pool and project: [B, T, f] -> [B, f] -> [B, h_dim]
            x = x.mean(dim=1)
            x = self.proj(x)
        elif self.encoder_type == "mlp":
            # [B, T, f] -> mean pool -> [B, f] -> [B, h_dim]
            x = x.mean(dim=1)
            x = self.encoder(x)
        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}")
        
        return x


class FusionModule(nn.Module):
    """
    Fusion module for combining multiple feature representations.
    
    Input: List of [B, h_dim] tensors
    Output: [B, new_h_dim] fused features
    """
    
    def __init__(
        self,
        input_dims: List[int],  # List of input dimensions
        output_dim: int,
        fusion_type: str = "concat_mlp",  # "concat_mlp", "add", "attention", "bilinear"
        **fusion_kwargs
    ):
        """
        Initialize fusion module.
        
        Args:
            input_dims: List of input dimensions for each modality
            output_dim: Output dimension (new_h_dim)
            fusion_type: Type of fusion ("concat_mlp", "add", "attention", "bilinear")
            **fusion_kwargs: Additional arguments for fusion
        """
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.fusion_type = fusion_type
        self.num_modalities = len(input_dims)
        
        if fusion_type == "concat_mlp":
            # Concatenate and pass through MLP
            input_dim = sum(input_dims)
            self.fusion = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, output_dim),
                nn.GELU(),
                nn.Linear(output_dim, output_dim)
            )
        elif fusion_type == "add":
            # Add features (requires same input dimensions)
            if not all(d == input_dims[0] for d in input_dims):
                raise ValueError("Add fusion requires all input dimensions to be equal")
            self.projs = nn.ModuleList([nn.Linear(d, output_dim) for d in input_dims])
            self.fusion = nn.Identity()
        elif fusion_type == "attention":
            # Multi-modal attention
            if not all(d == input_dims[0] for d in input_dims):
                raise ValueError("Attention fusion requires all input dimensions to be equal")
            query_dim = input_dims[0]
            self.query = nn.Linear(query_dim, output_dim)
            self.keys = nn.ModuleList([nn.Linear(d, output_dim) for d in input_dims])
            self.values = nn.ModuleList([nn.Linear(d, output_dim) for d in input_dims])
            self.attention = nn.MultiheadAttention(output_dim, num_heads=fusion_kwargs.get("num_heads", 4), batch_first=True)
            self.fusion = nn.Linear(output_dim, output_dim)
        elif fusion_type == "bilinear":
            # Bilinear fusion (for 2 modalities)
            if len(input_dims) != 2:
                raise ValueError("Bilinear fusion requires exactly 2 modalities")
            self.fusion = nn.Bilinear(input_dims[0], input_dims[1], output_dim)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: List of [B, h_dim] tensors
        
        Returns:
            Fused features [B, new_h_dim]
        """
        if self.fusion_type == "concat_mlp":
            # Concatenate: List of [B, h_dim] -> [B, sum(h_dims)]
            fused = torch.cat(features, dim=1)
            # MLP: [B, sum(h_dims)] -> [B, new_h_dim]
            fused = self.fusion(fused)
        
        elif self.fusion_type == "add":
            # Project each: [B, h_dim] -> [B, new_h_dim]
            projected = [proj(f) for proj, f in zip(self.projs, features)]
            # Add: List of [B, new_h_dim] -> [B, new_h_dim]
            fused = sum(projected)
            fused = self.fusion(fused)
        
        elif self.fusion_type == "attention":
            # Stack and use attention
            # Project to same dimension
            keys = torch.stack([k(f) for k, f in zip(self.keys, features)], dim=1)  # [B, M, new_h_dim]
            values = torch.stack([v(f) for v, f in zip(self.values, features)], dim=1)  # [B, M, new_h_dim]
            query = self.query(features[0]).unsqueeze(1)  # [B, 1, new_h_dim]
            
            # Attention: [B, 1, new_h_dim]
            fused, _ = self.attention(query, keys, values)
            fused = fused.squeeze(1)  # [B, new_h_dim]
            fused = self.fusion(fused)
        
        elif self.fusion_type == "bilinear":
            # Bilinear: [B, h_dim1] × [B, h_dim2] -> [B, new_h_dim]
            fused = self.fusion(features[0], features[1])
        
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")
        
        return fused


class PVInsightModel(LightningModule):
    """
    PVInsight: Generalist architecture for solar irradiance forecasting.
    
    Configurable components:
    - Video encoder: Processes image sequences
    - Irradiance encoder: Processes time-series features
    - Fusion module: Combines modalities
    - Prediction head: Generates forecasts
    """
    
    def __init__(
        self,
        # Video encoder config
        video_input_shape: Tuple[int, int, int] = (3, 448, 448),
        num_frames: int = 30,
        video_h_dim: int = 256,
        video_temporal_pooling: str = "mean",  # "mean", "last", "none"
        video_backbone: str = "simvp",
        
        # Irradiance encoder config
        irradiance_features: int = 6,
        irradiance_timesteps: Optional[int] = 6,
        irradiance_h_dim: int = 256,
        irradiance_encoder_type: str = "tcn",  # "tcn", "lstm", "transformer", "mlp"
        irradiance_num_layers: int = 2,
        
        # Fusion config
        fusion_output_dim: int = 512,
        fusion_type: str = "concat_mlp",  # "concat_mlp", "add", "attention", "bilinear"
        
        # Prediction head config
        output_channels: int = 2,
        prediction_horizons: int = 6,
        head_hidden_dim: int = 256,
        
        # Training config
        learning_rate: float = 3e-4,
        weight_decay: float = 0.05,
        loss_beta: float = 0.05,
        
        **kwargs
    ):
        """
        Initialize PVInsight model.
        
        Args:
            video_input_shape: Input image shape (C, H, W)
            num_frames: Number of input frames
            video_h_dim: Video encoder output dimension
            video_temporal_pooling: Temporal pooling type
            video_backbone: Video encoder backbone name
            
            irradiance_features: Number of irradiance features
            irradiance_timesteps: Number of irradiance timesteps
            irradiance_h_dim: Irradiance encoder output dimension
            irradiance_encoder_type: Irradiance encoder type
            irradiance_num_layers: Number of encoder layers
            
            fusion_output_dim: Fusion output dimension
            fusion_type: Fusion type
            
            output_channels: Number of output channels
            prediction_horizons: Number of prediction horizons
            head_hidden_dim: Prediction head hidden dimension
            
            learning_rate: Learning rate
            weight_decay: Weight decay
            loss_beta: SmoothL1Loss beta parameter
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        # Video encoder
        self.video_encoder = VideoEncoder(
            input_shape=video_input_shape,
            num_frames=num_frames,
            h_dim=video_h_dim,
            temporal_pooling=video_temporal_pooling,
            backbone=video_backbone,
            **kwargs
        )
        
        # Irradiance encoder
        self.irradiance_encoder = IrradianceEncoder(
            input_features=irradiance_features,
            h_dim=irradiance_h_dim,
            num_timesteps=irradiance_timesteps,
            encoder_type=irradiance_encoder_type,
            num_layers=irradiance_num_layers,
            **kwargs
        )
        
        # Fusion module
        input_dims = [video_h_dim, irradiance_h_dim]
        self.fusion = FusionModule(
            input_dims=input_dims,
            output_dim=fusion_output_dim,
            fusion_type=fusion_type,
            **kwargs
        )
        
        # Prediction head (simple MLP for now)
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(fusion_output_dim),
            nn.Linear(fusion_output_dim, head_hidden_dim),
            nn.GELU(),
            nn.Linear(head_hidden_dim, head_hidden_dim),
            nn.GELU(),
            nn.Linear(head_hidden_dim, output_channels * prediction_horizons)
        )
        
        # Loss function
        self.criterion = nn.SmoothL1Loss(beta=loss_beta, reduction="mean")
        
        # Store hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.output_channels = output_channels
        self.prediction_horizons = prediction_horizons
    
    def forward(
        self,
        images: torch.Tensor,
        irradiance: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            images: Input images [B, T, C, H, W]
            irradiance: Irradiance features [B, T_irr, f]
        
        Returns:
            Predictions [B, output_channels, prediction_horizons]
        """
        # Encode video: [B, T, C, H, W] -> [B, T_out, video_h_dim]
        video_features = self.video_encoder(images)
        
        # Encode irradiance: [B, T_irr, f] -> [B, irradiance_h_dim]
        # Note: Dataset returns [B, channels, timesteps], but encoder expects [B, timesteps, features]
        # So we transpose: [B, channels, timesteps] -> [B, timesteps, channels]
        if irradiance.dim() == 3 and irradiance.shape[1] == irradiance.shape[2]:
            # Assume dataset format is [B, channels, timesteps], transpose to [B, timesteps, channels]
            irradiance = irradiance.transpose(1, 2)  # [B, channels, timesteps] -> [B, timesteps, channels]
        irradiance_features = self.irradiance_encoder(irradiance)
        
        # Prepare for fusion (ensure video features are [B, video_h_dim])
        if video_features.dim() == 3:  # [B, T_out, h_dim]
            if video_features.size(1) == 1:
                video_features = video_features.squeeze(1)  # [B, h_dim]
            else:
                video_features = video_features.mean(dim=1)  # [B, h_dim]
        
        # Fuse: List of [B, h_dim] -> [B, fusion_output_dim]
        fused = self.fusion([video_features, irradiance_features])
        
        # Predict: [B, fusion_output_dim] -> [B, output_channels * prediction_horizons]
        predictions = self.prediction_head(fused)
        
        # Reshape: [B, output_channels * prediction_horizons] -> [B, output_channels, prediction_horizons]
        predictions = predictions.view(-1, self.output_channels, self.prediction_horizons)
        
        return predictions
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        images = batch['images']
        irradiance = batch['irradiance']
        target = batch['target']
        
        # Forward pass
        predictions = self(images, irradiance)
        
        # Compute loss
        loss = self.criterion(predictions, target)
        
        # Logging
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        images = batch['images']
        irradiance = batch['irradiance']
        target = batch['target']
        
        # Forward pass
        predictions = self(images, irradiance)
        
        # Compute loss
        loss = self.criterion(predictions, target)
        
        # Compute metrics
        with torch.no_grad():
            mae = torch.nn.functional.l1_loss(predictions, target)
            mse = torch.nn.functional.mse_loss(predictions, target)
            rmse = torch.sqrt(mse)
        
        # Logging
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val/mae', mae, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val/rmse', rmse, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return optimizer
