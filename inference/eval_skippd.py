#!/usr/bin/env python3
"""
Evaluation script for SKIPPD residual forecasting model.

Usage:
    python eval_skippd.py --checkpoint <checkpoint_path> [options]
"""

import importlib.util
import sys
from pathlib import Path

import torch
import numpy as np
import argparse

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

_skippd_path = _project_root / "datasets" / "skippd.py"
_spec = importlib.util.spec_from_file_location("skippd_module", _skippd_path)
_skippd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_skippd)
SkippdDataset = _skippd.SkippdDataset

from models import SkippdModel


def load_checkpoint(checkpoint_path, model, device):
    """Load model checkpoint. Handles DDP state dict (removes 'module.' prefix)."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {
            k[7:] if k.startswith("module.") else k: v
            for k, v in state_dict.items()
        }
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    return checkpoint


def evaluate_sample(model, sample, device, criterion):
    """Evaluate a single sample. Returns pred, target, loss, mae, rmse."""
    with torch.no_grad():
        images = sample["image"].unsqueeze(0).to(device)
        residual = sample["residual"].unsqueeze(0).to(device)
        cos_zenith = sample["cos_zenith"].unsqueeze(0).to(device)
        cos_azimuth = sample["cos_azimuth"].unsqueeze(0).to(device)
        sin_azimuth = sample["sin_azimuth"].unsqueeze(0).to(device)
        target = sample["target"].flatten()[0:1].to(device)  # single scalar; use first if multiple

        tabular = torch.stack(
            [residual, cos_zenith],
            # [residual, cos_zenith, cos_azimuth, sin_azimuth],
            dim=1,
        )

        pred = model(images, tabular)    # prediction of residual
        # use preidction of residual and pv_cs_target to predict pv_target
        pv_target = pred * 5.6 - 2.8 + sample["pv_cs_target"].to(device)
        pv_p = pv_target.cpu().numpy()[0]
        pv_t = sample["pv_target"].cpu().numpy()[0]
        mae_pv = float(np.abs(pv_p - pv_t))
        rmse_pv = float(np.sqrt((pv_p - pv_t) ** 2))

        loss = criterion(pred, target)

        p = pred.cpu().numpy()[0]
        t = target.cpu().numpy()[0]
        mae = float(np.abs(p - t))
        rmse = float(np.sqrt((p - t) ** 2))

        return {
            "prediction": p,
            "target": t,
            "prediction_pv": pv_p,
            "target_pv": pv_t,
            "loss": loss.item(),
            "mae": mae,
            "rmse": rmse,
            "mae_pv": mae_pv,
            "rmse_pv": rmse_pv,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SKIPPD residual forecasting model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval_skippd.py --checkpoint checkpoints/skippd_training/checkpoint_epoch_20.pth
  python eval_skippd.py --checkpoint checkpoints/skippd_training/checkpoint_final.pth --num_samples 1000 --output results/skippd_eval.npy
        """,
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--alpha", type=float, default=0.02717255163581315, help="Alpha for test pv_cs (default: 0.027..)")
    parser.add_argument("--lon", type=float, default=-122.174, help="Longitude (degrees)")
    parser.add_argument("--lat", type=float, default=34.427, help="Latitude (degrees)")
    parser.add_argument("--num_samples", type=int, default=None, help="Max test samples (default: all)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output", type=str, default=None, help="Path to save predictions/targets (.npy)")

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print("Creating model...")
    model = SkippdModel(image_size=64, num_frames=15, hidden_dim=256).to(device)
    print(f"Loading checkpoint: {ckpt_path}")
    load_checkpoint(ckpt_path, model, device)

    print("\nCreating test dataset...")
    test_dataset = SkippdDataset(
        lon=args.lon,
        lat=args.lat,
        split="test",
        alpha=args.alpha,
    )
    n_total = len(test_dataset)
    if args.num_samples is not None:
        n_eval = min(n_total, args.num_samples)
    else:
        n_eval = n_total
    print(f"Test dataset size: {n_total}, evaluating: {n_eval}")

    criterion = torch.nn.SmoothL1Loss(reduction="mean")
    model.eval()

    print("\nStarting evaluation...")
    print("-" * 60)
    all_results = []

    for idx in range(n_eval):
        sample = test_dataset[idx]
        result = evaluate_sample(model, sample, device, criterion)
        all_results.append(result)
        if (idx + 1) % 100 == 0 or (idx + 1) == n_eval:
            print(f"Processed {idx + 1}/{n_eval}  Loss: {result['loss']:.6f}  MAE: {result['mae']:.6f}  MAE_PV: {result['mae_pv']:.6f}  RMSE_PV: {result['rmse_pv']:.6f}")

    preds = np.array([r["prediction"] for r in all_results])
    targets = np.array([r["target"] for r in all_results])
    loss_mean = np.mean([r["loss"] for r in all_results])
    mae_mean = np.mean(np.abs(preds - targets))
    rmse_mean = np.sqrt(np.mean((preds - targets) ** 2))

    preds_pv = np.array([r["prediction_pv"] for r in all_results])
    targets_pv = np.array([r["target_pv"] for r in all_results])
    mae_pv_mean = np.mean(np.abs(preds_pv - targets_pv))
    rmse_pv_mean = np.sqrt(np.mean((preds_pv - targets_pv) ** 2))

    print("-" * 60)
    print(f"Summary  N={n_eval}  Loss: {loss_mean:.6f}  MAE: {mae_mean:.6f}  RMSE: {rmse_mean:.6f}  MAE_PV: {mae_pv_mean:.6f}  RMSE_PV: {rmse_pv_mean:.6f}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(
            out_path.with_suffix(".npy"),
            {
                "predictions": preds,
                "targets": targets,
                "mae": mae_mean,
                "rmse": rmse_mean,
                "loss_mean": loss_mean,
            },
        )
        print(f"\nResults saved to {out_path.with_suffix('.npy')}")


if __name__ == "__main__":
    main()
