from pathlib import Path
import sys
import yaml
import torch
from datetime import datetime as dt
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

sys.path.append(str(Path(__file__).parent.parent))
from datasets.folsom_day_ahead import FolsomDayAheadDataModule
from models.day_ahead_model import DayAhead
from training.callbacks import WandbImageSanityCallback, WandbGradNormCallback


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')

    data_config = {
        "root_dir": "/mnt/nfs/yuan/Folsom",
        # day-ahead: start with a single (target, horizon) so the dataset isn't empty after dropna filtering.
        # Once you move to masked training (handling missing horizons), you can switch these to None.
        "target": None,   # "ghi" | "dni" | ["ghi","dni"] | None
        "horizon": None,  # "26h" | ... | ["26h","27h"] | None
        "feature_return": "structured",
        "train_sample_num": None,
        "val_sample_num": None,
        "batch_size": 8,
        "num_workers": 16,
        "use_cache": True,
        "cache_dir": None,
        "drop_last": False,
    }
    model_config = {
        "class_path": "models.day_ahead_model.DayAhead",
        "num_targets": 2,
        "num_horizons": 14,
        "hidden_dim": 256,
        "depth": 3,
        "dropout": 0.1,
        "kt_max": 1.2,
    }
    training_config = {
        "learning_rate": 3e-4,
        "weight_decay": 0.05,
        "loss_beta": 0.05,
        "num_epochs": 1000,
        "seed": 42,
        "deterministic": True,
    }
    lightning_config = {
        "accelerator": "gpu",
        "devices": -1,
        "strategy": "ddp",
        "precision": 32,
        "accumulate_grad_batches": 1,
        "gradient_clip_val": None,
        "max_steps": None,
        "val_check_interval": 1.0,
        "log_every_n_steps": 50,
        "limit_train_batches": 1.0,
        "limit_val_batches": 1.0,
    }
    checkpoint_config = {
        "checkpoint_dir": None,  # set after save_dir is created
        "resume_from_checkpoint": None,
        "save_top_k": 3,
        "monitor": "val_irr_rmse_ghi",
        "mode": "min",
    }
    wandb_config = {
        "project": "ai4energy-folsom",
        "name": "day_ahead_forecasting",
        "entity": None,
    }
    experiment_config = {
        "run_name": "day_ahead_mlp",
    }

    config = {
        "data": data_config,
        "model": model_config,
        "training": training_config,
        "lightning": lightning_config,
        "checkpoint": checkpoint_config,
        "wandb": wandb_config,
        "experiment": experiment_config,
    }
    
    seed_everything(config["training"]["seed"], workers=True)
    
    datamodule = FolsomDayAheadDataModule(**config["data"])
    datamodule.setup("fit")

    # Derive dims from the dataset so shapes always match
    endo_dim = len(datamodule.train_dataset.feature_cols_endo)
    num_targets = len(datamodule.train_dataset.targets)
    num_horizons = len(datamodule.train_dataset.horizons)
    target_names = [str(t) for t in datamodule.train_dataset.targets]

    print("train_dataset len:", len(datamodule.train_dataset))
    print("val_dataset len:", len(datamodule.val_dataset))
    print("num_targets:", num_targets, "num_horizons:", num_horizons, "endo_dim:", endo_dim)

    if len(datamodule.train_dataset) == 0:
        raise RuntimeError(
            "train_dataset length is 0. If you're using target=None and horizon=None, "
            "the dataset dropna filtering can become very strict across all targets+horizons. "
            "Try training with a single target/horizon first, or change the dataset to support masking."
        )
        
    model_kwargs = {
        **{k: v for k, v in config["model"].items() if k != "class_path"},
        "num_targets": num_targets,
        "num_horizons": num_horizons,
        "endo_dim": endo_dim,
        "target_names": target_names,
        "learning_rate": config["training"]["learning_rate"],
        "weight_decay": config["training"]["weight_decay"],
        "loss_beta": config["training"]["loss_beta"],
    }
    model = DayAhead(**model_kwargs)
    
    timestamp = dt.now().strftime("%B-%d-%Y-%I-%M-%S-%p")
    run_name = config["experiment"]["run_name"]
    save_dir = Path(__file__).parent.parent / "runs" / run_name / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    config["experiment"]["timestamp"] = timestamp
    config["experiment"]["save_dir"] = str(save_dir)
    config["checkpoint"]["checkpoint_dir"] = str(save_dir)
    
    # Save config to yaml
    config_path = save_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="epoch_{epoch:02d}-val_loss_{val_loss:.4f}",
        auto_insert_metric_name=False,
        monitor=config["checkpoint"]["monitor"],
        mode=config["checkpoint"]["mode"],
        save_top_k=config["checkpoint"]["save_top_k"],
        save_last=True,
    )

    early_stopping = EarlyStopping(
        monitor=config["checkpoint"]["monitor"],
        mode=config["checkpoint"]["mode"],
        patience=100,
    )
    
    callbacks = [checkpoint_callback]
    callbacks += [
        early_stopping,
        WandbImageSanityCallback(),
        WandbGradNormCallback(),
    ]
    
    logger = WandbLogger(
        project=config["wandb"]["project"],
        name=config["wandb"]["name"],
        entity=config["wandb"]["entity"],
        log_model=False,  # Set to True if you want to log model checkpoints to wandb
    )
    
    # Create trainer
    trainer_kwargs = {
        "max_epochs": config["training"]["num_epochs"],
        "accelerator": config["lightning"]["accelerator"],
        "devices": config["lightning"]["devices"],
        "strategy": config["lightning"]["strategy"],
        "precision": config["lightning"]["precision"],
        "accumulate_grad_batches": config["lightning"]["accumulate_grad_batches"],
        "val_check_interval": config["lightning"]["val_check_interval"],
        "log_every_n_steps": config["lightning"]["log_every_n_steps"],
        "limit_train_batches": config["lightning"]["limit_train_batches"],
        "limit_val_batches": config["lightning"]["limit_val_batches"],
        "callbacks": callbacks,
        "logger": logger,
        "deterministic": config["training"]["deterministic"],
        "enable_progress_bar": True,
        "enable_model_summary": True,
    }
    
    # Only add max_steps if it's not None
    if config["lightning"]["max_steps"] is not None:
        trainer_kwargs["max_steps"] = config["lightning"]["max_steps"]
    
    # Only add gradient_clip_val if it's not None
    if config["lightning"]["gradient_clip_val"] is not None:
        trainer_kwargs["gradient_clip_val"] = config["lightning"]["gradient_clip_val"]
    
    trainer = Trainer(**trainer_kwargs)
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=config["checkpoint"]["resume_from_checkpoint"]
    )
    
    # Final evaluation on the test split (logs to the same WandB run)
    trainer.test(
        model=None,
        datamodule=datamodule,
        ckpt_path="best",
    )

    print(f"\nTraining completed!")
    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best model score: {checkpoint_callback.best_model_score}")
