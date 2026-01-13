"""
PyTorch Lightning training script for intra-hour forecasting model.

This script follows the same logic as training_folsom_mgpu.py but uses
PyTorch Lightning for simplified multi-GPU training.

Usage:
    # Single GPU
    python training/train.py
    
    # Multi-GPU (using Lightning's built-in support)
    # Set gpus parameter in __main__ block
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import pickle
import hashlib
import yaml
from datetime import datetime as dt
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

# Add parent directory to path to import datasets and models
sys.path.append(str(Path(__file__).parent.parent))
from datasets.folsom import FolsomDataset
from models import PVInsightModel


def get_cache_key(root_dir, split, sample_num, image_size):
    """Generate a cache key based on dataset parameters."""
    key_string = f"{root_dir}_{split}_{sample_num}_{image_size}"
    return hashlib.md5(key_string.encode()).hexdigest()


def load_or_create_dataset(
    root_dir,
    split,
    sample_num,
    image_size,
    cache_dir,
    use_cache=True
):
    """
    Load dataset from cache or create new one.
    
    Args:
        root_dir: Root directory of the dataset
        split: Dataset split ("train" or "test")
        sample_num: Number of samples
        image_size: Image size
        cache_dir: Directory to store cache files
        use_cache: Whether to use cache (default: True)
    
    Returns:
        FolsomDataset instance
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_key = get_cache_key(root_dir, split, sample_num, image_size)
    cache_file = cache_dir / f"dataset_{split}_{cache_key}.pkl"
    
    if use_cache and cache_file.exists():
        print(f"Loading {split} dataset from cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                dataset = pickle.load(f)
            print(f"Successfully loaded cached {split} dataset ({len(dataset)} samples)")
            return dataset
        except Exception as e:
            print(f"Warning: Failed to load cache ({e}), creating new dataset...")
    
    # Create new dataset
    print(f"Creating {split} dataset (this may take a while)...")
    dataset = FolsomDataset(
        root_dir=root_dir,
        split=split,
        sample_num=sample_num,
        image_size=image_size
    )
    
    # Save to cache
    if use_cache:
        print(f"Saving {split} dataset to cache: {cache_file}")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(dataset, f)
            print(f"Successfully cached {split} dataset")
        except Exception as e:
            print(f"Warning: Failed to save cache ({e})")
    
    return dataset


def create_data_loaders(
    root_dir,
    train_sample_num,
    val_sample_num,
    batch_size,
    num_workers,
    image_size,
    cache_dir=None,
    use_cache=True
):
    """
    Create training and validation data loaders with caching support.
    
    Args:
        root_dir: Root directory of the dataset
        train_sample_num: Number of training samples
        val_sample_num: Number of validation samples
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Image size
        cache_dir: Directory for cache files (default: .cache/datasets)
        use_cache: Whether to use cache (default: True)
    """
    # Set default cache directory
    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent / ".cache" / "datasets"
    else:
        cache_dir = Path(cache_dir)
    
    # Load or create datasets with caching
    train_dataset = load_or_create_dataset(
        root_dir=root_dir,
        split="train",
        sample_num=train_sample_num,
        image_size=image_size,
        cache_dir=cache_dir,
        use_cache=use_cache
    )
    
    val_dataset = load_or_create_dataset(
        root_dir=root_dir,
        split="test",  # Using test split as validation
        sample_num=val_sample_num,
        image_size=image_size,
        cache_dir=cache_dir,
        use_cache=use_cache
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    print(f"  Batch size per GPU: {batch_size}")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # ============ Data Parameters ============
    root_dir = "/mnt/nfs/yuan/Folsom"
    train_sample_num = 50000
    val_sample_num = 5000
    batch_size = 4  # Per-GPU batch size (reduced from 8 to avoid CUDA OOM)
    num_workers = 8  # Per-GPU workers
    
    # ============ Model Parameters ============
    image_size = 448
    num_frames = 30
    output_channels = 2
    hidden_dim = 256
    dropout = 0.1  # Note: Currently not used in PVInsight, reserved for future use
    
    # ============ Training Parameters ============
    learning_rate = 3e-4
    weight_decay = 0.05
    loss_beta = 0.05
    num_epochs = 10
    
    # ============ Scheduler Parameters ============
    # Note: PVInsightModel doesn't support scheduler yet, reserved for future use
    use_scheduler = False
    scheduler_type = "cosine"  # "cosine", "step", or "plateau"
    warmup_epochs = 0
    
    # ============ Lightning Parameters ============
    # For PyTorch Lightning 2.0+: use accelerator and devices instead of gpus
    accelerator = "gpu"  # "gpu", "cpu", or "auto"
    devices = -1  # -1 = use all available devices, or specify number like 4, or list like [0, 1, 2, 3]
    strategy = "ddp_find_unused_parameters_true"  # "ddp", "ddp_find_unused_parameters_true", "ddp_spawn", or "deepspeed"
    precision = 32  # 16 or 32
    accumulate_grad_batches = 1
    gradient_clip_val = None
    max_steps = None
    val_check_interval = 1.0
    log_every_n_steps = 1  # Log every step for more frequent logging
    limit_train_batches = 1.0
    limit_val_batches = 1.0
    
    # ============ Checkpointing and Logging ============
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints" / "folsom_lightning"
    resume_from_checkpoint = None  # Path to checkpoint to resume from
    save_top_k = 3
    monitor = "val/loss"
    mode = "min"  # "min" or "max"
    
    # ============ WandB Parameters ============
    wandb_project = "ai4energy-folsom"
    wandb_name = "intra_hour_forecasting"
    wandb_entity = None  # Your wandb entity/team name, or None
    
    # ============ Experiment Saving Parameters ============
    run_name = "timesformer_lstm_attention"  # Name for this experiment run
    
    # ============ Other Parameters ============
    seed = 42
    deterministic = True # (true for reproducibility but slower)
    
    # ============ Caching Parameters ============
    use_cache = True  # Set to False to disable caching
    cache_dir = None  # None = use default (.cache/datasets), or specify custom path
    
    # Set random seed for reproducibility
    seed_everything(seed, workers=True)
    
    # Create data loaders with caching
    train_loader, val_loader = create_data_loaders(
        root_dir=root_dir,
        train_sample_num=train_sample_num,
        val_sample_num=val_sample_num,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        cache_dir=cache_dir,
        use_cache=use_cache
    )
    
    # Calculate effective batch size
    if devices == -1:
        # When using -1, PL will use all available GPUs, so we'll calculate after trainer init
        # For now, use batch_size (will be updated after we know actual device count)
        effective_batch_size = batch_size
    elif isinstance(devices, int) and devices > 0:
        effective_batch_size = batch_size * devices
    elif isinstance(devices, (list, tuple)) and len(devices) > 0:
        effective_batch_size = batch_size * len(devices)
    else:
        effective_batch_size = batch_size
    
    # Create model (PVInsightModel) - using TimeSformer, LSTM, and Attention fusion
    model = PVInsightModel(
        # Video encoder config - using TimeSformer
        video_input_shape=(3, image_size, image_size),
        num_frames=num_frames,
        video_h_dim=hidden_dim,  # 256
        video_temporal_pooling="mean",  # Temporal pooling
        video_backbone="timesformer",  # Using TimeSformer instead of SimVP
        
        # Irradiance encoder config - using LSTM
        irradiance_features=6,  # From dataset: 6 channels
        irradiance_timesteps=6,  # From dataset: 6 timesteps
        irradiance_h_dim=hidden_dim,  # 256
        irradiance_encoder_type="lstm",  # Using LSTM instead of TCN
        irradiance_num_layers=2,  # Number of LSTM layers
        
        # Fusion config - using attention
        fusion_output_dim=hidden_dim * 2,  # 512
        fusion_type="attention",  # Using attention fusion instead of concat_mlp
        
        # Prediction head config
        output_channels=output_channels,  # 2
        prediction_horizons=6,  # 6 time horizons (5min, 10min, ..., 30min)
        head_hidden_dim=hidden_dim,  # 256
        
        # Training config
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        loss_beta=loss_beta,
    )
    
    # Setup experiment saving with timestamp
    timestamp = dt.now().strftime("%B-%d-%Y-%I-%M-%S-%p")
    save_dir = Path(__file__).parent.parent / "runs" / run_name / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create config dictionary for saving
    config = {
        "data": {
            "root_dir": str(root_dir),
            "train_sample_num": train_sample_num,
            "val_sample_num": val_sample_num,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "image_size": image_size,
        },
        "model": {
            "video_backbone": "timesformer",
            "video_h_dim": hidden_dim,
            "num_frames": num_frames,
            "irradiance_encoder_type": "lstm",
            "irradiance_h_dim": hidden_dim,
            "fusion_type": "attention",
            "fusion_output_dim": hidden_dim * 2,
            "output_channels": output_channels,
            "prediction_horizons": 6,
        },
        "training": {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "loss_beta": loss_beta,
            "num_epochs": num_epochs,
        },
        "lightning": {
            "accelerator": accelerator,
            "devices": devices if devices != -1 else "all",
            "strategy": strategy,
            "precision": precision,
            "log_every_n_steps": log_every_n_steps,
        },
        "wandb": {
            "project": wandb_project,
            "name": wandb_name,
            "entity": wandb_entity,
        },
        "other": {
            "seed": seed,
            "deterministic": deterministic,
        }
    }
    
    # Save config to yaml
    config_path = save_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="epoch_{epoch:02d}-val_loss_{val/loss:.4f}",
        monitor=monitor,
        mode=mode,
        save_top_k=save_top_k,
        save_last=True,
        verbose=True
    )
    
    callbacks = [checkpoint_callback]
    
    # Add learning rate monitor if using scheduler (not supported in PVInsightModel yet)
    # if use_scheduler:
    #     lr_monitor = LearningRateMonitor(logging_interval="step")
    #     callbacks.append(lr_monitor)
    
    # Create WandB logger
    logger = WandbLogger(
        project=wandb_project,
        name=wandb_name,
        entity=wandb_entity,
        log_model=False,  # Set to True if you want to log model checkpoints to wandb
    )
    
    # Determine if multi-device training
    num_devices = None
    if devices == -1:
        # -1 means use all available devices, so we'll assume multi-device
        use_multi_device = True
    elif isinstance(devices, int) and devices > 0:
        num_devices = devices
        use_multi_device = num_devices > 1
    elif isinstance(devices, (list, tuple)) and len(devices) > 0:
        num_devices = len(devices)
        use_multi_device = num_devices > 1
    else:
        use_multi_device = False
    
    # Create trainer
    trainer_kwargs = {
        "max_epochs": num_epochs,
        "accelerator": accelerator,
        "devices": devices,
        "strategy": strategy if use_multi_device else "auto",
        "precision": precision,
        "accumulate_grad_batches": accumulate_grad_batches,
        "val_check_interval": val_check_interval,
        "log_every_n_steps": log_every_n_steps,
        "limit_train_batches": limit_train_batches,
        "limit_val_batches": limit_val_batches,
        "callbacks": callbacks,
        "logger": logger,
        "deterministic": deterministic,
        "enable_progress_bar": True,
        "enable_model_summary": True,
    }
    
    # Only add max_steps if it's not None
    if max_steps is not None:
        trainer_kwargs["max_steps"] = max_steps
    
    # Only add gradient_clip_val if it's not None
    if gradient_clip_val is not None:
        trainer_kwargs["gradient_clip_val"] = gradient_clip_val
    
    trainer = Trainer(**trainer_kwargs)
    
    print(f"\nStarting training...")
    print(f"  Experiment: {run_name}")
    print(f"  Timestamp: {timestamp}")
    print(f"  Save directory: {save_dir}")
    print(f"  Config saved to: {config_path}")
    print(f"  WandB Project: {wandb_project}")
    print(f"  WandB Run: {wandb_name}")
    print(f"  Accelerator: {accelerator}")
    print(f"  Devices: {devices if devices != -1 else 'all available'}")
    print(f"  Strategy: {strategy if use_multi_device else 'auto'}")
    print(f"  Precision: {precision}-bit")
    print(f"  Epochs: {num_epochs}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Logging every {log_every_n_steps} steps")
    
    # Train the model
    # In PL 2.0+, ckpt_path should be passed to fit(), not Trainer constructor
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_from_checkpoint  # Pass checkpoint path to fit() method
    )
    
    print(f"\nTraining completed!")
    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best model score: {checkpoint_callback.best_model_score}")
