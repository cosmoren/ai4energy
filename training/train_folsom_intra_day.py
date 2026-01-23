import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from pathlib import Path
import sys

# Add parent directory to path to import datasets
sys.path.append(str(Path(__file__).parent.parent))
from datasets.folsom_intra_day import FolsomIntraDayDataset
from models.intra_day_model import intra_day_model
from evaluation.evaluation import Evaluation

def evaluate_on_test(model, test_loader, device):
    model.eval()

    all_preds = []
    all_timestamps = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["images"].to(device)
            irradiance = batch["irradiance"].to(device)
            timestamps = batch["timestamp"]

            outputs = model(
                images=images,
                irradiance=irradiance,
            )  # [B, 2, 6]

            all_preds.append(outputs)
            all_timestamps.extend(timestamps)

    pred_tensor = torch.cat(all_preds, dim=0)  # [N, 2, 6]
    pred_tensor = pred_tensor.cpu().numpy()

    evaluator = Evaluation()

    df_ghi = evaluator.eval(
        eval_type="intra-day",
        target="ghi",
        model_name="intra_day",
        result=pred_tensor[:, 0, :],  # [N, 6]
    )
    df_dni = evaluator.eval(
        eval_type="intra-day",
        target="dni",
        model_name="intra_day",
        result=pred_tensor[:, 1, :],  # [N, 6]
    )

    return df_ghi, df_dni

def log_eval_to_tensorboard(writer, df, prefix, epoch):
    # 单 horizon
    for i, row in df.iterrows():
        horizon = row["horizon"]
        writer.add_scalar(
            f"{prefix}/RMSE_{horizon}",
            row["intra_day_rmse"],
            epoch,
        )

    # 平均 RMSE
    writer.add_scalar(
        f"{prefix}/RMSE_mean",
        df["intra_day_rmse"].mean(),
        epoch,
    )

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
    train_dataset = FolsomIntraDayDataset(root_dir=root_dir, split="train", sample_num=50000)
    
    # print("Creating test dataset...")
    # test_dataset = FolsomIntraDayDataset(root_dir=root_dir, split="test", sample_num=5000)
    
    # Create data loaders
    batch_size = 64
    num_workers = 32
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print("Creating test dataset...")
    test_dataset = FolsomIntraDayDataset(root_dir=root_dir, split="test",)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,          # evaluation 用 1，和你现在 test 脚本一致
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_dataset)} images")
    # print(f"  Test: {len(test_dataset)} images")
    print(f"  Batch size: {batch_size}")
    print(f"  Training batches: {len(train_loader)}")
    
    model = intra_day_model(
        image_size = 10,
        num_frames = 3,
        num_channels = 2,
    ).to(device)
    criterion = torch.nn.SmoothL1Loss(beta=0.05, reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.05, eps=1e-8) # lr = 3e-4
    
    # Initialize TensorBoard writer
    log_dir = Path(__file__).parent.parent / "runs" / "folsom_intra_day_training"
    writer = SummaryWriter(log_dir=log_dir)
    
    # Create checkpoints directory
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints" / "folsom_intra_day_training"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    
    num_epochs = 400
    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()  # Zero gradients before each backward pass
            
            images = batch['images'] 
            irradiance = batch['irradiance']
            target = batch['target']
            
            # check for NaN
            if (
                torch.isnan(images).any()
                or torch.isnan(irradiance).any()
                or torch.isnan(target).any()
            ):
                print("NaN found in input. Skipping")
                continue

            outputs = model(images.to(device), irradiance.to(device))
            loss = criterion(outputs, target.to(device))
            loss.backward()
            optimizer.step()

            # Write loss to TensorBoard
            writer.add_scalar('Loss/Train', loss.item(), global_step)
            global_step += 1

            if global_step % 50 == 0:
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

        # ========== Evaluation on test set ==========
        df_ghi, df_dni = evaluate_on_test(
            model=model,
            test_loader=test_loader,
            device=device,
        )
        log_eval_to_tensorboard(writer, df_ghi, "Test/GHI", epoch)
        log_eval_to_tensorboard(writer, df_dni, "Test/DNI", epoch)
        print(f"[Epoch {epoch}] Test evaluation done")

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
