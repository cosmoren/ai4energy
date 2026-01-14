from pathlib import Path
import sys

import numpy as np
import torch
from pytorch_lightning import Trainer
from rich.console import Console
from rich.table import Table

# allow running as a script from repo root
sys.path.append(str(Path(__file__).parent.parent))
from datasets.lightning import FolsomDataModule
from evaluation.evaluation import Evaluation
from models.pvinsight import PVFormer


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    checkpoint_path = "/mnt/nfs/slurm/home/mohammed2/with_yuan/ai4energy/runs/timesformer_lstm_attention/January-14-2026-09-26-03-PM/epoch_epoch=09-val_loss_val/loss=0.0562.ckpt"

    root_dir = "/mnt/nfs/yuan/Folsom"
    image_size = 224
    batch_size = 16
    num_workers = 8

    out_path = Path(checkpoint_path).parent / "infer_results.npy"

    if out_path.exists():
        preds = np.load(out_path)
    else:
        model = PVFormer.load_from_checkpoint(checkpoint_path)

        dm = FolsomDataModule(
            root_dir=root_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            train_sample_num=1,
            val_sample_num=1,
            test_sample_num=None,
            use_cache=True,
        )
        dm.setup("test")

        trainer = Trainer(accelerator="auto", devices=1, logger=False, enable_checkpointing=False)
        preds = trainer.predict(model, dataloaders=dm.test_dataloader())

        preds = torch.cat([p.detach().cpu() for p in preds], dim=0).numpy()
        np.save(out_path, preds)

    print("preds shape:", preds.shape)  # [N, 2, 6]
    print("saved/loaded:", str(out_path))

    evaluator = Evaluation(base_dir=root_dir)
    ghi_df = evaluator.eval(eval_type="intra-hour", target="ghi", model_name="pvformer", result=preds[:, 0, :])
    dni_df = evaluator.eval(eval_type="intra-hour", target="dni", model_name="pvformer", result=preds[:, 1, :])

    console = Console()
    def show_df(title: str, df):
        table = Table(title=title, header_style="bold")
        for c in df.columns:
            table.add_column(str(c))
        def fmt(x):
            if x is None:
                return ""
            if isinstance(x, (float, int, np.floating, np.integer)):
                if np.isnan(x):
                    return "nan"
                return f"{float(x):.3f}"
            return str(x)
        for _, row in df.iterrows():
            table.add_row(*[fmt(x) for x in row.values])
        console.print(table)

    show_df("GHI", ghi_df)
    show_df("DNI", dni_df)

