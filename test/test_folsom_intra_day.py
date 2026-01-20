import numpy as np
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_dataset = FolsomIntraDayDataset(
    root_dir="/mnt/nfs/yuan/Folsom",
    split="test",
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False, # test, no shuffle
    num_workers=4,
    pin_memory=True if torch.cuda.is_available() else False,
)

model = intra_day_model(
    image_size=10,
    num_frames=3,
    num_channels=1,
).to(device)

ckpt_path = Path(
    "/mnt/nfs/slurm/home/weize2/folsom/ai4energy/checkpoints/folsom_intra_day_training/checkpoint_epoch_25.pth"
)

ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])

model.eval()

all_preds = []
all_timestamps = []

with torch.no_grad():
    for batch in test_loader:
        images = batch["images"].to(device)         # [B, 3, 1, 10, 10]
        irradiance = batch["irradiance"].to(device) # [B, 6, 6]
        timestamps = batch["timestamp"]

        outputs = model(
            images=images,
            irradiance=irradiance,
        )  # [B, 2, 6]
        all_preds.append(outputs)
        all_timestamps.extend(timestamps)

pred_tensor = torch.cat(all_preds, dim=0).squeeze(dim=1) # [8068, 2, 6]
pred_tensor = pred_tensor.cpu().numpy()

# eval
evaluator = Evaluation()
df_ghi = evaluator.eval(
    eval_type="intra-day",
    target="ghi",
    model_name="intra_day_20",
    result=pred_tensor[:, 0, :], # shape [N, 6]
)
df_dni = evaluator.eval(
    eval_type="intra-day",
    target="dni",
    model_name="intra_day_20",
    result=pred_tensor[:, 1, :], # shape [N, 6]
)
print("**** GHI ****")
print(df_ghi)
print("**** DNI ****")
print(df_dni)
#df_ghi.to_csv("/mnt/nfs/slurm/home/weize2/folsom/ai4energy/ghi_intra_day_ckpt20.csv")
#df_dni.to_csv("/mnt/nfs/slurm/home/weize2/folsom/ai4energy/dni_intra_day_ckpt20.csv")
#breakpoint()