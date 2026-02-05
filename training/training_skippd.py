"""
Training script for SKIPPD residual forecasting.
Uses SkippdDataset and SkippdModel (SimVP + TCN + fusion, same structure as intra_hour_model).
"""

import importlib.util
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

_skippd_path = _project_root / "datasets" / "skippd.py"
_spec = importlib.util.spec_from_file_location("skippd_module", _skippd_path)
_skippd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_skippd)
SkippdDataset = _skippd.SkippdDataset

from models import SkippdModel


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Creating training dataset...")
    train_dataset = SkippdDataset(
        lon=-122.174,
        lat=34.427,
        split="train",
        sample_num=100000,
    )

    batch_size = 32
    num_workers = 8
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Train samples: {len(train_dataset)}, batches: {len(train_loader)}")

    model = SkippdModel(
        image_size=64,
        num_frames=15,
        hidden_dim=256,
    ).to(device)

    criterion = torch.nn.SmoothL1Loss(reduction="mean")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
        betas=(0.9, 0.95),
        weight_decay=0.05,
        eps=1e-8,
    )

    log_dir = Path(__file__).resolve().parent.parent / "runs" / "skippd_training"
    writer = SummaryWriter(log_dir=log_dir)

    ckpt_dir = Path(__file__).resolve().parent.parent / "checkpoints" / "skippd_training"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints: {ckpt_dir}")

    num_epochs = 20
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            images = batch["image"]
            residual = batch["residual"]
            cos_zenith = batch["cos_zenith"]
            cos_azimuth = batch["cos_azimuth"]
            sin_azimuth = batch["sin_azimuth"]
            target = batch["target"].squeeze(-1)

            tabular = torch.stack(
                [residual, cos_zenith],
                # [residual, cos_zenith, cos_azimuth, sin_azimuth],
                dim=1,
            )

            pred = model(images.to(device), tabular.to(device))
            loss = criterion(pred, target.to(device))
            loss.backward()
            optimizer.step()
            writer.add_scalar("Loss/Train", loss.item(), global_step)
            global_step += 1

        ckpt_path = ckpt_dir / f"checkpoint_epoch_{epoch + 1}.pth"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            ckpt_path,
        )
        print(f"Checkpoint saved: {ckpt_path}")

    final_path = ckpt_dir / "checkpoint_final.pth"
    torch.save(
        {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
        },
        final_path,
    )
    print(f"Final checkpoint saved: {final_path}")
    writer.close()


if __name__ == "__main__":
    main()
