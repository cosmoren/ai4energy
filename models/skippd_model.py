"""
SKIPPD intra-hour model: video + tabular fusion for residual regression.
Structure mirrors intra_hour_model (SimVP + TCN + fusion) but adapted for SKIPPD:
- 15 frames, 64×64 images
- Tabular: residual, cos_zenith, cos_azimuth, sin_azimuth over 15 steps → [B, 4, 15]
- Output: single scalar per sample (residual at t+15min).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import sys
from pathlib import Path

models_dir = Path(__file__).parent
if str(models_dir) not in sys.path:
    sys.path.insert(0, str(models_dir))

from .simvp.models import SimVP_Model
from .intra_hour_model import TemporalConvPool, SmallTCN


class SmallMLP(nn.Module):
    """MLP for 1D sequence: [B, D, T] -> [B, D_enc]. Flattens input, no temporal modeling."""

    def __init__(self, D: int, D_enc: int = 256, T: int = 15):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(D * T, D_enc),
            nn.GELU(),
            nn.Linear(D_enc, D_enc),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, T = x.shape
        x = x.view(B, -1)
        return self.mlp(x)


class RegressionHead(nn.Module):
    """Maps fused [B, z_dim] to scalar [B] for residual prediction."""

    def __init__(self, z_dim: int, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(z_dim),
            nn.Linear(z_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z).squeeze(-1)


class SkippdModel(nn.Module):
    """
    SKIPPD model: SimVP video encoder + TCN on tabular series → fusion → scalar residual.

    Inputs:
        - images: [B, 15, 3, H, W] (H,W typically 64)
        - tabular: [B, 4, 15] (residual, cos_zenith, cos_azimuth, sin_azimuth)
    Output: [B] predicted residual at t+15min
    """

    def __init__(
        self,
        image_size: int = 64,
        num_frames: int = 15,
        video_embed_dim: int = 1024,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.video_embed_dim = video_embed_dim
        self.hidden_dim = hidden_dim
        self.image_size = image_size

        self.simvp = SimVP_Model([num_frames, 3, image_size, image_size])
        self.temporal_conv_pool = TemporalConvPool(16, hidden_dim)
        self.pv_encoder = SmallTCN(D=1, D_enc=256)
        self.zenith_encoder = SmallMLP(D=1, D_enc=256)
        self.fusion_head = RegressionHead(z_dim=hidden_dim * 3, hidden=hidden_dim)

    def forward(
        self,
        images: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            images: [B, T, C, H, W], T=15, C=3
            tabular: [B, 4, T] (residual, cos_zenith, cos_azimuth, sin_azimuth)
        Returns:
            [B] predicted residual
        """
        B, T, C, H, W = images.shape

        x = images.view(B * T, C, H, W)
        embed, skip = self.simvp.enc(x)
        _, C_, H_, W_ = embed.shape
        z = embed.view(B, T, C_, H_, W_)
        hid = self.simvp.hid(z)
        video_features = hid.mean(dim=[3, 4])
        video_encoded = self.temporal_conv_pool(video_features)

        pv_encoded = self.pv_encoder(tabular[:, 0, :][:, None, :])
        zenith_encoded = self.zenith_encoder(tabular[:, 1, :][:, None, :])
        fused = torch.cat([video_encoded, pv_encoded, zenith_encoded], dim=1)
        out = self.fusion_head(fused)
        return out
