import torch
import torch.nn as nn
from typing import Optional
import sys
from pathlib import Path

# Add models directory to path so simvp can be imported as a top-level module
# This allows simvp files to use "from simvp.xxx" imports without modification
models_dir = Path(__file__).parent
if str(models_dir) not in sys.path:
    sys.path.insert(0, str(models_dir))

from .simvp.models import SimVP_Model

class TemporalConvPool(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size=3, padding=1, dilation=1),
            nn.GELU(),
            nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=4, dilation=4),
            nn.GELU(),
        )

    def forward(self, x):
        """
        x: [B, T, C]
        return: [B, DIM]
        """
        x = x.transpose(1, 2)   # [B, C, T]
        x = self.net(x)         # [B, DIM, T]
        x = x.mean(dim=-1)      # temporal mean pooling → [B, DIM]
        return x


class SmallTCN(nn.Module):
    def __init__(self, D, D_enc=256):
        super().__init__()
        self.in_proj = nn.Conv1d(D, D_enc, 1)
        self.conv = nn.Sequential(
            nn.Conv1d(D_enc, D_enc, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(D_enc, D_enc, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, x):        # [B,D,T]
        x = self.in_proj(x)      # [B,D_enc,T]
        x = self.conv(x)         # [B,D_enc,T]
        return x.mean(dim=-1)    # [B,D_enc]



class HorizonHead(nn.Module):
    def __init__(self, z_dim=512, h_dim=64, hidden=256, out_channels=2, T=6):
        super().__init__()
        self.T = T
        self.out_channels = out_channels

        self.fuse = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, z_dim),
            nn.GELU(),
        )

        self.h_emb = nn.Embedding(T, h_dim)

        self.mlp = nn.Sequential(
            nn.LayerNorm(z_dim + h_dim),
            nn.Linear(z_dim + h_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_channels),
        )

    def forward(self, z):
        B = z.shape[0]

        h = torch.arange(self.T, device=z.device)           # [T]
        e = self.h_emb(h).unsqueeze(0).expand(B, -1, -1)    # [B,T,h_dim]
        z_rep = z.unsqueeze(1).expand(-1, self.T, -1)       # [B,T,z_dim]

        inp = torch.cat([z_rep, e], dim=-1)                 # [B,T,z_dim+h_dim]
        y = self.mlp(inp)                                   # [B,T,2]
        # Apply sigmoid to get [0, 1] range, then scale to [0, 1.2]
        y = torch.sigmoid(y) * 1.2                          # [B,T,2] -> [0, 1.2]
        return y.permute(0, 2, 1).contiguous()              # [B,2,T]

class intra_hour_model(nn.Module):
    """
    Complete model for intra-hour forecasting.
    Combines VideoTransformer with irradiance feature processing.
    
    Inputs:
        - images: [B, 30, 3, 448, 448]
        - irradiance: [B, irradiance_dim]
    Output: [B, target_dim] or [B, 30, target_dim] depending on task
    """
    
    def __init__(
        self,
        image_size: int = 448,
        num_frames: int = 30,
        video_embed_dim: int = 1024,
        output_channels: int = 2,  # Will be inferred from data
        hidden_dim: int = 256,
        dropout: float = 0.1,
        **kwargs
    ):
        """
        Args:
            image_size: Size of input images (default: 448)
            num_frames: Number of frames (default: 30)
            video_embed_dim: Embedding dimension from VideoTransformer (default: 1024)
            irradiance_dim: Dimension of irradiance features (will be inferred if None)
            target_dim: Dimension of target output (will be inferred if None)
            hidden_dim: Hidden dimension for fusion layers (default: 512)
            dropout: Dropout rate (default: 0.1)
            **kwargs: Additional arguments for VideoTransformer
        """
        super().__init__()
        self.num_frames = num_frames
        self.video_embed_dim = video_embed_dim
        self.output_channels = output_channels
        self.hidden_dim = hidden_dim
        self.image_size = image_size

        self.simvp = SimVP_Model([num_frames, 3, image_size, image_size])
        self.temporal_conv_pool = TemporalConvPool(16, hidden_dim)
        
        # Irradiance feature processing
        self.irradiance_encoder = SmallTCN(D=6, D_enc=hidden_dim)
        
        # Define a custom scaling layer for [0, 1.2] range
        class ScaleLayer(nn.Module):
            def __init__(self, scale=1.2):
                super().__init__()
                self.scale = scale
            
            def forward(self, x):
                return x * self.scale
        
        self.fusion_layer = HorizonHead(z_dim=hidden_dim*2, h_dim=64, hidden=hidden_dim, out_channels=output_channels, T=6)
    
    def forward(
        self, 
        images: torch.Tensor, 
        irradiance: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            images: Input tensor of shape [B, T, C, H, W]
                   B = batch size
                   T = number of frames (30)
                   C = channels (3)
                   H, W = image size (224, 224)
            irradiance: Optional irradiance features of shape [B, irradiance_dim]
        
        Returns:
            Output tensor of shape [B, target_dim]
        """
        B, T, C, H, W = images.shape
        
        # Normalize images to [0, 1] range if they're in [0, 255]
        if images.max() > 1.0:
            images = images / 255.0
        
        # Reshape for SimVP encoder: [B, T, C, H, W] -> [B*T, C, H, W]
        x = images.view(B * T, C, H, W)
        
        # Process through SimVP encoder
        embed, skip = self.simvp.enc(x)  # embed: [B*T, C_, H_, W_], skip: list of skip connections
        _, C_, H_, W_ = embed.shape
        # Reshape encoder output: [B*T, C_, H_, W_] -> [B, T, C_, H_, W_]
        z = embed.view(B, T, C_, H_, W_)
        # SimVP translator: self.hid (MidMetaNet or MidIncepNet)
        hid = self.simvp.hid(z)  # [B, T, C_, H_, W_]
    
        # [B, T, C_, H_, W_] - pool spatial dimensions to get per-frame features
        video_features = hid.mean(dim=[3, 4])  # [B, T, C_]
        video_encoded = self.temporal_conv_pool(video_features)  # [B, hidden_dim]
    
        irradiance_encoded = self.irradiance_encoder(irradiance)  # [B, hidden_dim]


        # Fuse video and irradiance features
        fused = torch.cat([video_encoded, irradiance_encoded], dim=1)  # [B, 2*hidden_dim]

        # Final fusion and processing
        output = self.fusion_layer(fused)  # [B, hidden_dim]
        
        return output


