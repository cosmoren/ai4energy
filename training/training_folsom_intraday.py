import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from pathlib import Path
import sys

# Add parent directory to path to import datasets
sys.path.append(str(Path(__file__).parent.parent))
from datasets.folsom_intraday import FolsomIntradayDataset


class SmallTCN(nn.Module):
    def __init__(self, D, D_enc=128):
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

def masked_mean(x, mask, dim=1, eps=1e-6):
    # x: [B,T,D], mask: [B,T] bool/0-1
    w = mask.float().unsqueeze(-1)            # [B,T,1]
    return (x * w).sum(dim=dim) / (w.sum(dim=dim).clamp_min(eps))

class SatEncoderTCN(nn.Module):
    def __init__(self, in_dim=100, emb_dim=128, hidden=128):
        super().__init__()
        self.frame_mlp = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace=True),
        )
        # temporal conv over T
        self.tcn = nn.Sequential(
            nn.Conv1d(emb_dim, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Linear(hidden, hidden)

    def forward(self, x, mask):
        """
        x:    [B, T, 100]   (0-filled for missing)
        mask: [B, T] bool   (True=valid, False=missing)
        return: [B, hidden]
        """
        B, T, D = x.shape

        # per-frame embedding
        h = self.frame_mlp(x)                 # [B,T,emb_dim]

        # help Conv1d ignore missing frames:
        # multiply by mask so missing frames become true zeros in feature space
        h = h * mask.float().unsqueeze(-1)    # [B,T,emb_dim]

        # temporal conv expects [B, C, T]
        h = self.tcn(h.transpose(1, 2)).transpose(1, 2)  # [B,T,hidden]

        # masked pooling over time
        z = masked_mean(h, mask, dim=1)       # [B,hidden]
        return self.out(z)

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

class IntraDayModel(nn.Module):
    """
    Model for intra-day forecasting.
    
    Inputs:
        - irradiance: [B, 6, 6] - 6 time horizons, 6 features (B, V, L for ghi_kt and dni_kt)
        - satellite_features: [B, 12, 100] - 12 timesteps, 100 features
        - satellite_features_mask: [B, 12] - mask for valid timesteps
    Output: [B, 2, 6] - 2 dimensions (ghi_kt, dni_kt), 6 time horizons (30min, 60min, 90min, 120min, 150min, 180min)
    """
    
    def __init__(
        self,
        satellite_dim: int = 100,
        satellite_timesteps: int = 12,
        hidden_dim: int = 128,
        num_horizons: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()
        self.satellite_dim = satellite_dim
        self.satellite_timesteps = satellite_timesteps
        self.hidden_dim = hidden_dim
        self.num_horizons = num_horizons
        
        # Irradiance feature encoder: flatten and process
        self.irradiance_encoder = SmallTCN(D=6, D_enc=hidden_dim)

        # Satellite features encoder: process temporal sequence
        self.satellite_encoder = SatEncoderTCN(in_dim=satellite_dim, emb_dim=hidden_dim, hidden=hidden_dim)
        
        # Fusion layer
        self.fusion_layer = HorizonHead(z_dim=hidden_dim*2, h_dim=64, hidden=hidden_dim, out_channels=2, T=6)
        
    def forward(
        self,
        irradiance: torch.Tensor,
        satellite_features: torch.Tensor,
        satellite_features_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            irradiance: [B, 6, 6]
            satellite_features: [B, 12, 100]
            satellite_features_mask: [B, 12] - True for valid, False for padding
        
        Returns:
            [B, 2, 6] - predictions for 6 time horizons
        """
        B = irradiance.shape[0]
        
        # Process irradiance features
        irradiance_encoded = self.irradiance_encoder(irradiance)  # [B, hidden_dim, T]
        
        # Process satellite features
        satellite_encoded = self.satellite_encoder(satellite_features, satellite_features_mask.float())
        
        fused = torch.cat([irradiance_encoded, satellite_encoded], dim=1)

        output = self.fusion_layer(fused)  # [B, hidden_dim]

        return output


def main():
    """
    Main training function for Folsom intra-day forecasting.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset paths
    root_dir = "/mnt/nfs/yuan/Folsom"
    
    # Create datasets
    print("Creating training dataset...")
    train_dataset = FolsomIntradayDataset(root_dir=root_dir, split="train", sample_num=10000)
    
    print("Creating test dataset...")
    test_dataset = FolsomIntradayDataset(root_dir=root_dir, split="test")
    
    # Create data loaders
    batch_size = 16
    num_workers = 8
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    print(f"  Batch size: {batch_size}")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Create model
    model = IntraDayModel().to(device)
    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Loss and optimizer
    criterion = nn.SmoothL1Loss(beta=0.05, reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=0.01, eps=1e-8) 
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    
    # Initialize TensorBoard writer
    log_dir = Path(__file__).parent.parent / "runs" / "folsom_intraday_training"
    writer = SummaryWriter(log_dir=log_dir)
    
    # Create checkpoints directory
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints" / "folsom_intraday_training"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    
    num_epochs = 100
    global_step = 0
    best_test_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        num_train_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            irradiance = batch['irradiance'].to(device)  # [B, 6, 6]
            satellite_features = batch['satellite_features'].to(device)  # [B, 12, 100]
            satellite_features_mask = batch['satellite_features_mask'].to(device)  # [B, 12]
            target = batch['target'].to(device)  # [B, 2, 6]
                  
            outputs = model(irradiance, satellite_features, satellite_features_mask)  # [B, 2, 6]
            loss = criterion(outputs, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            num_train_batches += 1
            global_step += 1
            
            # Write loss to TensorBoard
            writer.add_scalar('Loss/Train', loss.item(), global_step)
            
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.6f}')
        
        avg_train_loss = train_loss / num_train_batches
        writer.add_scalar('Loss/Train_epoch', avg_train_loss, epoch)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Evaluation phase
        model.eval()
        test_loss = 0.0
        num_test_batches = 0
        
        with torch.no_grad():
            for batch in test_loader:
                irradiance = batch['irradiance'].to(device)
                satellite_features = batch['satellite_features'].to(device)
                satellite_features_mask = batch['satellite_features_mask'].to(device)
                target = batch['target'].to(device)
                outputs = model(irradiance, satellite_features, satellite_features_mask)
                loss = criterion(outputs, target)
                
                if ~torch.isnan(target).any():
                    test_loss += loss.item()
                    num_test_batches += 1
        
        avg_test_loss = test_loss / num_test_batches
        writer.add_scalar('Loss/Test', avg_test_loss, epoch)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}, LR: {current_lr:.2e}')
        
        # Save checkpoint at the end of each epoch
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'global_step': global_step,
            'train_loss': avg_train_loss,
            'test_loss': avg_test_loss,
        }, checkpoint_path)
        
        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_checkpoint_path = checkpoint_dir / "checkpoint_best.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'global_step': global_step,
                'train_loss': avg_train_loss,
                'test_loss': avg_test_loss,
            }, best_checkpoint_path)
            print(f"New best model saved with test loss: {best_test_loss:.6f}")
    
    # Save final checkpoint
    final_checkpoint_path = checkpoint_dir / "checkpoint_final.pth"
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'global_step': global_step,
        'train_loss': avg_train_loss,
        'test_loss': avg_test_loss,
    }, final_checkpoint_path)
    print(f"Final checkpoint saved: {final_checkpoint_path}")
    
    # Close TensorBoard writer
    writer.close()
    print("Training completed!")


if __name__ == "__main__":
    main()
