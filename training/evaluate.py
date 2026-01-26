from pathlib import Path
import sys
import importlib

import numpy as np
import torch
from pytorch_lightning import Trainer
from rich.console import Console
from rich.table import Table

sys.path.append(str(Path(__file__).parent.parent))
import yaml
from datasets.folsom_intra_hour import FolsomDataModule
from evaluation.evaluation import Evaluation


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    checkpoint_path = ""

    ckpt_path = Path(checkpoint_path)
    config_path_candidates = [
        ckpt_path.parent / "config.yaml",
        ckpt_path.parent.parent / "config.yaml",
    ]
    config_path = next((p for p in config_path_candidates if p.exists()), None)
    if config_path is None:
        raise FileNotFoundError(
            "Could not find config.yaml next to checkpoint. Tried: "
            + ", ".join(str(p) for p in config_path_candidates)
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    class_path = config["model"]["class_path"]
    module_name, class_name = class_path.rsplit(".", 1)
    ModelCls = getattr(importlib.import_module(module_name), class_name)

    root_dir = config["data"]["root_dir"]
    image_size = config["data"]["image_size"]
    batch_size = 16
    num_workers = 8

    out_path = Path(checkpoint_path).parent / "infer_results.npy"

    if out_path.exists():
        preds = np.load(out_path)
    else:
        model = ModelCls.load_from_checkpoint(checkpoint_path)

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

        trainer = Trainer(accelerator="auto", devices=1, logger=False, enable_checkpointing=False) # devices must be = 1 for validity
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

