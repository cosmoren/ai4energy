from pathlib import Path
import sys
import yaml
import torch
from datetime import datetime as dt
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

sys.path.append(str(Path(__file__).parent.parent))
from datasets.lightning import FolsomDataModule
from models.pvinsight import PVFormer


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')

    image_size = 224
    data_config = {
        "root_dir": "/mnt/nfs/yuan/Folsom",
        "train_sample_num": 100000,
        "val_sample_num": 100000,
        "batch_size": 4,
        "num_workers": 16,
        "image_size": image_size,
        "use_cache": True,
        "cache_dir": None,
    }
    model_config = {
        "image_size": image_size,
        "num_frames": 30,
        "output_channels": 2,
        "hidden_dim": 256,
    }
    training_config = {
        "learning_rate": 3e-4,
        "weight_decay": 0.05,
        "loss_beta": 0.05,
        "num_epochs": 10,
        "seed": 42,
        "deterministic": True,
    }
    lightning_config = {
        "accelerator": "gpu",
        "devices": -1,
        "strategy": "ddp_find_unused_parameters_true",
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
        "monitor": "val/loss",
        "mode": "min",
    }
    wandb_config = {
        "project": "ai4energy-folsom",
        "name": "intra_hour_forecasting",
        "entity": None,
    }
    experiment_config = {
        "run_name": "pvformer_intra_hour",
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
    
    datamodule = FolsomDataModule(**config["data"])
        
    model_kwargs = {
        **config["model"],
        "learning_rate": config["training"]["learning_rate"],
        "weight_decay": config["training"]["weight_decay"],
        "loss_beta": config["training"]["loss_beta"],
    }
    model = PVFormer(**model_kwargs)
    
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
        filename="epoch_{epoch:02d}-val_loss_{val/loss:.4f}",
        monitor=config["checkpoint"]["monitor"],
        mode=config["checkpoint"]["mode"],
        save_top_k=config["checkpoint"]["save_top_k"],
        save_last=True,
    )

    early_stopping = EarlyStopping(
        monitor=config["checkpoint"]["monitor"],
        mode=config["checkpoint"]["mode"],
        patience=5,
    )
    
    callbacks = [checkpoint_callback, early_stopping]
    
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
    
    print(f"\nTraining completed!")
    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best model score: {checkpoint_callback.best_model_score}")
