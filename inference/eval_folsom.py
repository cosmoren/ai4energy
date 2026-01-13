#!/usr/bin/env python3
"""
Evaluation script for Folsom intra-hour forecasting model.

Usage:
    python eval_folsom.py --checkpoint <checkpoint_path> [options]
"""

from ast import Break
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import argparse
import numpy as np
from collections import defaultdict

# Add parent directory to path to import datasets and models
sys.path.append(str(Path(__file__).parent.parent))
from datasets.folsom import FolsomDataset
from models import intra_hour_model


def load_checkpoint(checkpoint_path, model, device):
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model instance
        device: Device to load model to
    
    Returns:
        Dictionary with checkpoint information (epoch, etc.)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle both DDP and regular model state dicts
    state_dict = checkpoint['model_state_dict']
    
    # Remove 'module.' prefix if present (from DDP models)
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {key[7:] if key.startswith('module.') else key: value 
                     for key, value in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    return checkpoint


def evaluate_sample(model, sample, device, criterion):
    """
    Evaluate a single sample.
    
    Args:
        model: Model instance
        sample: Dictionary with 'images', 'irradiance', 'target', 'timestamp'
        device: Device to run evaluation on
        criterion: Loss function
    
    Returns:
        Dictionary with predictions, targets, loss, and metrics
    """
    with torch.no_grad():
        # Prepare inputs
        images = sample['images'].unsqueeze(0).to(device)  # Add batch dimension
        irradiance = sample['irradiance'].unsqueeze(0).to(device)  # Add batch dimension
        target = sample['target'].unsqueeze(0).to(device)  # Add batch dimension
        
        # Forward pass
        outputs =  model(images, irradiance)  # [B, 2, 6]
        
        # Calculate loss
        loss = criterion(outputs, target)
        
        # Convert to numpy for metrics
        pred = outputs.cpu().numpy()[0]  # [2, 6]
        targ = target.cpu().numpy()[0]  # [2, 6]
        
        # Calculate metrics
        mae = np.mean(np.abs(pred - targ))
        rmse = np.sqrt(np.mean((pred - targ) ** 2))
        
        # Per-channel metrics
        mae_ghi = np.mean(np.abs(pred[0] - targ[0]))
        mae_dni = np.mean(np.abs(pred[1] - targ[1]))
        rmse_ghi = np.sqrt(np.mean((pred[0] - targ[0]) ** 2))
        rmse_dni = np.sqrt(np.mean((pred[1] - targ[1]) ** 2))
        
        return {
            'timestamp': sample['timestamp'],
            'prediction': pred,
            'target': targ,
            'loss': loss.item(),
            'mae': mae,
            'rmse': rmse,
            'mae_ghi': mae_ghi,
            'mae_dni': mae_dni,
            'rmse_ghi': rmse_ghi,
            'rmse_dni': rmse_dni,
        }


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Folsom intra-hour forecasting model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with a specific checkpoint
  python eval_folsom.py --checkpoint checkpoints/folsom_training/checkpoint_epoch_10.pth
  
  # Evaluate with custom data directory
  python eval_folsom.py --checkpoint checkpoints/folsom_training/checkpoint_epoch_10.pth --data_dir /path/to/Folsom
  
  # Evaluate with specific number of test samples
  python eval_folsom.py --checkpoint checkpoints/folsom_training/checkpoint_epoch_10.pth --num_samples 1000
        """
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint file'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/mnt/nfs/yuan/Folsom',
        help='Root directory containing Folsom data (default: /mnt/nfs/yuan/Folsom)'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='Number of test samples to evaluate (default: all test samples)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use (default: cuda if available, else cpu)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save evaluation results and predictions (.npy file). Predictions include timestamps, GHI and DNI values for all samples.'
    )
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Create model
    print("Creating model...")
    model = intra_hour_model().to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint_info = load_checkpoint(checkpoint_path, model, device)
    
    # Create test dataset
    print(f"\nCreating test dataset from: {args.data_dir}")
    test_dataset = FolsomDataset(
        root_dir=args.data_dir,
        split="test",
        sample_num=args.num_samples if args.num_samples else 100000  # Use large number if None
    )
    
    if args.num_samples:
        # Limit dataset size if specified
        test_dataset.selected_keys = test_dataset.selected_keys[:args.num_samples]
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Loss function
    criterion = torch.nn.SmoothL1Loss(beta=0.05, reduction="mean")
    
    # Ensure model is in eval mode
    model.eval()
    
    # Evaluate samples one by one
    print("\nStarting evaluation...")
    print("-" * 80)
    
    all_results = []

    for idx in range(len(test_dataset)):
        sample = test_dataset[idx]
        result = evaluate_sample(model, sample, device, criterion)
        all_results.append(result)
        
        # Print progress
        if (idx + 1) % 100 == 0 or (idx + 1) == len(test_dataset):
            print(f"Processed {idx + 1}/{len(test_dataset)} samples "
                  f"(Loss: {result['loss']:.6f}, MAE: {result['mae']:.6f})")
    
    # Save predictions with timestamps
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract predictions and timestamps
        timestamps = [result['timestamp'] for result in all_results]
        predictions_ghi = np.array([result['prediction'][0] for result in all_results])  # [N, 6]
        predictions_dni = np.array([result['prediction'][1] for result in all_results])  # [N, 6]
        targets_ghi = np.array([result['target'][0] for result in all_results])  # [N, 6]
        targets_dni = np.array([result['target'][1] for result in all_results])  # [N, 6]
        
        # Save predictions with timestamps
        predictions_data = {
            'timestamps': np.array(timestamps, dtype=object),  # Array of timestamp strings
            'predictions_ghi': predictions_ghi,  # [N, 6] - GHI predictions for 6 time horizons
            'predictions_dni': predictions_dni,  # [N, 6] - DNI predictions for 6 time horizons
            'targets_ghi': targets_ghi,  # [N, 6] - GHI targets for 6 time horizons
            'targets_dni': targets_dni,  # [N, 6] - DNI targets for 6 time horizons
            'time_horizons': np.array([5, 10, 15, 20, 25, 30]),  # Time horizons in minutes
        }
        
        # Save as numpy file for easy loading
        np.save(output_path.with_suffix('.npy'), predictions_data)
        
        print(f"\nResults saved to: {output_path.with_suffix('.npy')}")
        print(f"  - Timestamps: {len(timestamps)} samples")
        print(f"  - GHI predictions shape: {predictions_ghi.shape}")
        print(f"  - DNI predictions shape: {predictions_dni.shape}")


if __name__ == "__main__":
    main()
