"""
End-to-end sanity check for FolsomIntraDayDataset.

This script replicates the official Forecast_intra-day.py training process:
- OLS / Ridge / Lasso
- Endogenous vs Exogenous features
- Feature normalization
- kt -> irradiance conversion
- elevation masking

Purpose:
Verify that the PyTorch dataloader is compatible with and can reproduce
the official intra-day forecasting pipeline.

This is NOT a unit test. It is an end-to-end replication sanity check.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader

# ---------------------------------------------------------------------
# Import dataset
# ---------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from datasets.folsom_intra_day import FolsomIntraDayDataset


ROOT = "/mnt/nfs/yuan/Folsom"


def run_sanity_check(target: str, horizon: str):
    print("\n" + "=" * 80)
    print(f"Intra-day sanity check | target={target}, horizon={horizon}")
    print("=" * 80)

    # -----------------------------------------------------------------
    # Load datasets
    # -----------------------------------------------------------------
    train_ds = FolsomIntraDayDataset(
        root_dir=ROOT,
        split="train",
        target=target,
        horizon=horizon,
    )
    test_ds = FolsomIntraDayDataset(
        root_dir=ROOT,
        split="test",
        target=target,
        horizon=horizon,
    )

    # deterministic loaders (no shuffle)
    train_loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)

    train_batch = next(iter(train_loader))
    test_batch = next(iter(test_loader))

    # -----------------------------------------------------------------
    # Extract arrays (match official variable names)
    # -----------------------------------------------------------------
    X_train = train_batch["features"].numpy()
    X_test = test_batch["features"].numpy()

    y_train = train_batch["target"].numpy()
    y_test = test_batch["target"].numpy()
    elev_train = train_batch["elevation"].numpy()
    elev_test = test_batch["elevation"].numpy()

    clear_train = train_batch["clear_sky"].numpy()
    clear_test = test_batch["clear_sky"].numpy()

    # endogenous features only
    train_ds_endo = train_ds.feature_cols_endo
    endo_idx = [train_ds.feature_cols.index(c) for c in train_ds_endo]

    X_train_endo = X_train[:, endo_idx]
    X_test_endo = X_test[:, endo_idx]

    # -----------------------------------------------------------------
    # Models (exactly as official)
    # -----------------------------------------------------------------
    models = [
        ("ols", linear_model.LinearRegression()),
        ("ridge", linear_model.RidgeCV(cv=10)),
        ("lasso", linear_model.LassoCV(cv=10, max_iter=10000)),
    ]

    for Xtr, Xte, feat_type in [
        (X_train_endo, X_test_endo, "endo"),
        (X_train, X_test, "exo"),
    ]:
        print(f"\n--- Feature set: {feat_type} ---")

        scaler = StandardScaler()
        scaler.fit(Xtr)

        Xtr = scaler.transform(Xtr)
        Xte = scaler.transform(Xte)

        for name, model in models:
            model.fit(Xtr, y_train)

            train_pred = model.predict(Xtr)
            test_pred = model.predict(Xte)

            # clip kt
            train_pred = np.clip(train_pred, 0, 1.1)
            test_pred = np.clip(test_pred, 0, 1.1)

            # convert kt -> irradiance
            train_pred *= clear_train
            test_pred *= clear_test

            # elevation mask
            train_pred[elev_train < 5] = np.nan
            test_pred[elev_test < 5] = np.nan

            rmse = np.sqrt(np.nanmean((test_pred - y_test * clear_test) ** 2))

            print(f"{name:6s} | RMSE = {rmse:8.2f} W/m^2")

    # -----------------------------------------------------------------
    # Smart persistence baseline (sp)
    # -----------------------------------------------------------------
    print("\n--- Baseline: smart persistence (sp) ---")

    # B(kt|30min) is part of endogenous features
    sp_idx = train_ds.feature_cols.index(f"B({target}_kt|30min)")

    sp_train = X_train[:, sp_idx] * clear_train
    sp_test = X_test[:, sp_idx] * clear_test

    sp_train[elev_train < 5] = np.nan
    sp_test[elev_test < 5] = np.nan

    rmse_sp = np.sqrt(np.nanmean((sp_test - y_test * clear_test) ** 2))
    print(f"sp     | RMSE = {rmse_sp:8.2f} W/m^2")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    targets = ["ghi", "dni"]
    horizons = ["30min", "60min", "90min", "120min", "150min", "180min"]

    for t in targets:
        for h in horizons:
            run_sanity_check(t, h)
