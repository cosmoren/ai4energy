import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from pathlib import Path
import sys
import os

# Add parent directory to path to import datasets
sys.path.append(str(Path(__file__).parent.parent))
from datasets.folsom import FolsomDataset
from models import intra_hour_model


def setup_distributed():
    """
    Initialize distributed training environment.
    This is called automatically by torchrun.
    """
    # torchrun sets these environment variables
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Initialize the process group
    dist.init_process_group(backend="nccl")
    
    # Set the device for this process
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    return rank, local_rank, world_size, device


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def main():
    """
    Main training function with multi-GPU support using torchrun.
    
    Usage:
        torchrun --nproc_per_node=4 training_folsom_mgpu.py
    """
    # Setup distributed training
    rank, local_rank, world_size, device = setup_distributed()
    
    # Only print on rank 0 to avoid duplicate output
    if rank == 0:
        print(f"Initializing distributed training with {world_size} GPUs")
        print(f"Rank {rank}, Local Rank {local_rank}, Device: {device}")
    
    # Dataset paths
    root_dir = "/mnt/nfs/yuan/Folsom"
    
    # Create datasets
    if rank == 0:
        print("Creating training dataset...")
    train_dataset = FolsomDataset(root_dir=root_dir, split="train", sample_num=50000)
    
    # Create data loaders with DistributedSampler
    batch_size = 8  # Per-GPU batch size
    num_workers = 8  # Adjust per GPU (32 / 4 GPUs = 8)
    
    # Create distributed sampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,  # Shuffle is handled by DistributedSampler
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch for consistent training
    )
    
    if rank == 0:
        print(f"\nDataset sizes:")
        print(f"  Training: {len(train_dataset)} images")
        print(f"  Batch size per GPU: {batch_size}")
        print(f"  Effective batch size: {batch_size * world_size}")
        print(f"  Training batches per GPU: {len(train_loader)}")
    
    # Create model and move to device
    model = intra_hour_model().to(device)
    
    # Wrap model with DDP
    # find_unused_parameters=True is needed when some parameters don't receive gradients
    # (e.g., due to conditional logic or unused model components)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    criterion = torch.nn.SmoothL1Loss(beta=0.05, reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.05, eps=1e-8)
    
    # Initialize TensorBoard writer (only on rank 0)
    writer = None
    checkpoint_dir = None
    if rank == 0:
        log_dir = Path(__file__).parent.parent / "runs" / "folsom_training_mgpu"
        writer = SummaryWriter(log_dir=log_dir)
        
        # Create checkpoints directory
        checkpoint_dir = Path(__file__).parent.parent / "checkpoints" / "folsom_training_mgpu"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"Checkpoints will be saved to: {checkpoint_dir}")
    
    num_epochs = 10
    global_step = 0
    
    for epoch in range(num_epochs):
        # Set epoch for DistributedSampler (important for shuffling)
        train_sampler.set_epoch(epoch)
        
        model.train()
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            images = batch['images']
            irradiance = batch['irradiance']
            target = batch['target']
            
            outputs = model(images.to(device), irradiance.to(device))
            loss = criterion(outputs, target.to(device))
            loss.backward()
            optimizer.step()
            
            # Write loss to TensorBoard (only on rank 0)
            if rank == 0:
                writer.add_scalar('Loss/Train', loss.item(), global_step)
                # Also log per-GPU if multiple GPUs
                if world_size > 1:
                    writer.add_scalar(f'Loss/Train_GPU_{rank}', loss.item(), global_step)
            
            global_step += 1
            
            # Print progress (only on rank 0)
            if rank == 0 and batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
        
        # Synchronize all processes before saving checkpoint
        dist.barrier()
        
        # Save checkpoint at the end of each epoch (only on rank 0)
        if rank == 0:
            # Save the underlying model (unwrap DDP)
            model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step,
                'world_size': world_size,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
        
        # Synchronize before next epoch
        dist.barrier()
    
    # Save final checkpoint (only on rank 0)
    if rank == 0:
        model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        
        final_checkpoint_path = checkpoint_dir / "checkpoint_final.pth"
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
            'world_size': world_size,
        }, final_checkpoint_path)
        print(f"Final checkpoint saved: {final_checkpoint_path}")
        
        # Close TensorBoard writer
        writer.close()
    
    # Cleanup distributed training
    cleanup_distributed()


if __name__ == "__main__":
    main()

