"""
Evaluation script for PVInsight model.

This script loads a pretrained PVInsight model checkpoint and evaluates it
using the evaluation framework from evaluation/evaluation.py.

Usage:
    # Update parameters in __main__ block, then run:
    python training/evaluate.py
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.model_summary import ModelSummary

# Add parent directory to path to import datasets and models
sys.path.append(str(Path(__file__).parent.parent))
from datasets.folsom import FolsomDataset
from models import PVInsightModel
from evaluation.evaluation import Evaluation


def load_or_create_test_dataset(
    root_dir,
    image_size,
    cache_dir=None,
    use_cache=True
):
    """Load or create test dataset (matching train.py logic)."""
    import pickle
    import hashlib
    
    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent / ".cache" / "datasets"
    else:
        cache_dir = Path(cache_dir)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create cache key (without sample_num since we use all available samples)
    cache_key = f"test_all_{image_size}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
    cache_path = cache_dir / f"dataset_{cache_hash}.pkl"
    
    if use_cache and cache_path.exists():
        print(f"Loading cached dataset from {cache_path}")
        with open(cache_path, "rb") as f:
            dataset = pickle.load(f)
    else:
        print(f"Creating dataset (cache: {use_cache})...")
        # Use a large sample_num to get all available test samples
        dataset = FolsomDataset(
            root_dir=root_dir,
            split="test",
            image_size=image_size
        )
        if use_cache:
            print(f"Saving dataset cache to {cache_path}")
            with open(cache_path, "wb") as f:
                pickle.dump(dataset, f)
    
    return dataset


def evaluate_model(
    checkpoint_path,
    root_dir="/mnt/nfs/yuan/Folsom",
    batch_size=16,
    num_workers=8,
    image_size=448,
    eval_type="intra-hour",
    target="ghi",
    model_name="pvinsight",
    device="cuda",
    cache_dir=None,
    use_cache=True
):
    """
    Evaluate a pretrained PVInsight model.
    
    Args:
        checkpoint_path: Path to the PyTorch Lightning checkpoint file
        root_dir: Root directory of the dataset
        batch_size: Batch size for evaluation
        num_workers: Number of data loading workers
        image_size: Image size
        eval_type: Evaluation type ("intra-hour", "intra-day", "day-ahead")
        target: Target variable ("ghi" or "dni")
        model_name: Name for the model in evaluation results
        device: Device to use ("cuda" or "cpu")
        cache_dir: Directory for cache files
        use_cache: Whether to use cached datasets
    
    Returns:
        pandas DataFrame with evaluation metrics
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"\nLoading model from checkpoint: {checkpoint_path}")
    
    # Load model from checkpoint
    # PyTorch Lightning's load_from_checkpoint automatically loads the model class
    model = PVInsightModel.load_from_checkpoint(str(checkpoint_path))
    model.eval()
    model = model.to(device)
    
    print("Model loaded successfully!")
    
    # Print model summary
    print("\nModel Summary:")
    summary = ModelSummary(model, max_depth=2)
    print(summary)
    
    # Load test dataset
    print(f"\nLoading test dataset...")
    test_dataset = load_or_create_test_dataset(
        root_dir=root_dir,
        image_size=image_size,
        cache_dir=cache_dir,
        use_cache=use_cache
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Run inference
    print(f"\nRunning inference on test set...")
    predictions_list = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Inference", unit="batch")):
            images = batch['images'].to(device)
            irradiance = batch['irradiance'].to(device)
            
            # Forward pass
            # Model output: [B, output_channels, prediction_horizons]
            # output_channels=2 (GHI, DNI), prediction_horizons=6
            pred = model(images, irradiance)
            
            # Convert to numpy and collect
            pred_np = pred.cpu().numpy()
            predictions_list.append(pred_np)
    
    # Concatenate all predictions
    all_predictions = np.concatenate(predictions_list, axis=0)
    # Shape: [N, output_channels, prediction_horizons]
    
    print(f"Predictions shape: {all_predictions.shape}")
    
    # Extract predictions for the target (GHI or DNI)
    # target_idx: 0 for GHI, 1 for DNI
    target_idx = 0 if target.lower() == "ghi" else 1
    target_predictions = all_predictions[:, target_idx, :]  # [N, prediction_horizons]
    
    print(f"Target predictions shape: {target_predictions.shape}")
    print(f"Target: {target.upper()}")
    
    # Initialize evaluator
    evaluator = Evaluation(base_dir=root_dir)
    
    # Run evaluation
    print(f"\nRunning evaluation...")
    results_df = evaluator.eval(
        eval_type=eval_type,
        target=target.lower(),
        model_name=model_name,
        result=target_predictions  # Shape: [N, 6] for intra-hour
    )
    
    print(f"\nEvaluation Results ({target.upper()}):")
    print(results_df.to_string())
    
    return results_df


if __name__ == "__main__":
    # ============ Checkpoint Parameters ============
    checkpoint_path = "/mnt/nfs/slurm/home/mohammed2/with_yuan/ai4energy/runs/timesformer_lstm_attention/January-13-2026-12-30-43-AM/epoch_epoch=09-val_loss_val/loss=0.0644.ckpt"  # Update this path
    
    # ============ Dataset Parameters ============
    root_dir = "/mnt/nfs/yuan/Folsom"
    batch_size = 16
    num_workers = 8
    image_size = 448
    
    # ============ Evaluation Parameters ============
    eval_type = "intra-hour"  # "intra-hour", "intra-day", or "day-ahead"
    target = "ghi"  # "ghi" or "dni"
    model_name = "pvinsight"
    
    # ============ Device Parameters ============
    device = "cuda"  # "cuda" or "cpu"
    
    # ============ Caching Parameters ============
    cache_dir = None  # None = use default (.cache/datasets), or specify custom path
    use_cache = True  # Set to False to disable caching
    
    # ============ Output Parameters ============
    output_path = None  # None = don't save, or specify path like "results/ghi_results.csv"
    
    # Run evaluation
    results_df = evaluate_model(
        checkpoint_path=checkpoint_path,
        root_dir=root_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        eval_type=eval_type,
        target=target,
        model_name=model_name,
        device=device,
        cache_dir=cache_dir,
        use_cache=use_cache
    )
    
    # Save results if output path is provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    
    print("\nEvaluation completed!")
