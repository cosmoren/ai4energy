import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from pathlib import Path
import sys

# Add parent directory to path to import datasets
sys.path.append(str(Path(__file__).parent.parent))
from datasets.folsom import FolsomDataset
from models import intra_hour_model


def main():
    """
    Main training function to test the Folsom dataloader.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset paths
    root_dir = "/mnt/nfs/yuan/Folsom"
    
    # Create datasets
    print("Creating training dataset...")
    train_dataset = FolsomDataset(root_dir=root_dir, split="train", sample_num=50000)
    
    # print("Creating test dataset...")
    # test_dataset = FolsomDataset(root_dir=root_dir, split="test", sample_num=5000)
    
    # Create data loaders
    batch_size = 8
    num_workers = 32
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    '''
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    '''
    
    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_dataset)} images")
    # print(f"  Test: {len(test_dataset)} images")
    print(f"  Batch size: {batch_size}")
    print(f"  Training batches: {len(train_loader)}")
    
    model = intra_hour_model().to(device)
    criterion = torch.nn.SmoothL1Loss(beta=0.05, reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.05, eps=1e-8)
    
    # Initialize TensorBoard writer
    log_dir = Path(__file__).parent.parent / "runs" / "folsom_training"
    writer = SummaryWriter(log_dir=log_dir)
    
    # Create checkpoints directory
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints" / "folsom_training"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    
    num_epochs = 10
    global_step = 0
    for epoch in range(num_epochs):
        # model.train()
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()  # Zero gradients before each backward pass
            
            images = batch['images'] 
            irradiance = batch['irradiance']
            target = batch['target']
            
            outputs = model(images.to(device), irradiance.to(device))
            loss = criterion(outputs, target.to(device))
            loss.backward()
            optimizer.step()

            # Write loss to TensorBoard
            writer.add_scalar('Loss/Train', loss.item(), global_step)
            global_step += 1

            print('loss: ', epoch, batch_idx, loss.item())
        
        # Save checkpoint at the end of each epoch
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final checkpoint
    final_checkpoint_path = checkpoint_dir / "checkpoint_final.pth"
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
    }, final_checkpoint_path)
    print(f"Final checkpoint saved: {final_checkpoint_path}")
    
    # Close TensorBoard writer
    writer.close()

if __name__ == "__main__":
    main()

