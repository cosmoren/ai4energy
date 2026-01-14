from pathlib import Path
from torch.utils.data import Dataset
from typing import Literal, Optional, Union, List
import pandas as pd
import random
import torch
import numpy as np


class FolsomIntraDayDataset(Dataset):
    """
    PyTorch Dataset for loading Folsom intra-day forecasting data.

    Matches official Forecast_intra-day.py logic:
    - Training set: years <= 2015
    - Test set: year == 2016

    Horizons: 30min, 60min, 90min, 120min, 150min, 180min

    Inputs:
    - Endogenous: Irradiance_features_intra-day.csv
    - Exogenous: Sat_image_features_intra-day.csv
    - Target: Target_intra-day.csv

    Output:
    - features: combined endogenous + satellite (exo) features
    - target: kt values
    - clear_sky: clear-sky irradiance
    - elevation: solar elevation
    - timestamp
    """

    ALL_HORIZONS = ["30min", "60min", "90min", "120min", "150min", "180min"]
    ALL_TARGETS = ["ghi", "dni"]

    def __init__(
        self,
        root_dir: str = "/mnt/nfs/yuan/Folsom",
        split: Literal["train", "test"] = "train",
        target: Optional[Union[str, List[str]]] = "ghi",
        horizon: Optional[Union[str, List[str]]] = "30min",
        sample_num: Optional[int] = None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split

        # -------------------------
        # Normalize targets
        # -------------------------
        if target is None:
            self.targets = self.ALL_TARGETS.copy()
        elif isinstance(target, str):
            self.targets = [target]
        else:
            self.targets = list(target)

        for t in self.targets:
            if t not in self.ALL_TARGETS:
                raise ValueError(f"Invalid target: {t}")
        self.targets = sorted(set(self.targets))

        # -------------------------
        # Normalize horizons
        # -------------------------
        if horizon is None:
            self.horizons = self.ALL_HORIZONS.copy()
        elif isinstance(horizon, str):
            self.horizons = [horizon]
        else:
            self.horizons = list(horizon)

        for h in self.horizons:
            if h not in self.ALL_HORIZONS:
                raise ValueError(f"Invalid horizon: {h}")
        self.horizons = sorted(set(self.horizons), key=lambda x: int(x[:-3]))

        # -------------------------
        # Load CSVs
        # -------------------------
        inpEndo = pd.read_csv(
            self.root_dir / "Irradiance_features_intra-day.csv",
            parse_dates=True,
            index_col=0,
        )
        inpExo = pd.read_csv(
            self.root_dir / "Sat_image_features_intra-day.csv",
            parse_dates=True,
            index_col=0,
        )
        tar = pd.read_csv(
            self.root_dir / "Target_intra-day.csv",
            parse_dates=True,
            index_col=0,
        )

        # -------------------------
        # Train / test split
        # -------------------------
        if split == "train":
            data = inpEndo[inpEndo.index.year <= 2015]
            data = data.join(inpExo[inpEndo.index.year <= 2015], how="inner")
            data = data.join(tar[tar.index.year <= 2015], how="inner")
        elif split == "test":
            data = inpEndo[inpEndo.index.year == 2016]
            data = data.join(inpExo[inpEndo.index.year == 2016], how="inner")
            data = data.join(tar[tar.index.year == 2016], how="inner")
        else:
            raise ValueError(f"Invalid split: {split}")

        # -------------------------
        # Required columns
        # -------------------------
        cols = []
        for t in self.targets:
            for h in self.horizons:
                cols.extend([
                    f"{t}_{h}",
                    f"{t}_kt_{h}",
                    f"{t}_clear_{h}",
                ])
        for h in self.horizons:
            cols.append(f"elevation_{h}")

        # -------------------------
        # Feature columns
        # -------------------------
        feature_cols_endo = []
        for t in self.targets:
            feature_cols_endo.extend(inpEndo.filter(regex=t).columns.tolist())
        feature_cols_endo = sorted(set(feature_cols_endo))

        feature_cols = feature_cols_endo + inpExo.columns.tolist()

        # -------------------------
        # Final data selection
        # -------------------------
        required_cols = list(dict.fromkeys(cols + feature_cols))
        data = data[required_cols].dropna(how="any")

        self.data = data
        self.feature_cols = feature_cols
        self.feature_cols_endo = feature_cols_endo

        self.indices = list(range(len(data)))

        if split == "train" and sample_num is not None:
            self.indices = random.sample(self.indices, min(sample_num, len(self.indices)))

        print(f"[Intra-day {split}] samples: {len(self.indices)}")
        print(f"  targets: {self.targets}")
        print(f"  horizons: {self.horizons}")
        print(f"  features: {len(self.feature_cols)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row = self.data.iloc[self.indices[idx]]

        features = torch.tensor(
            row[self.feature_cols].values.astype(np.float32)
        )

        # kt targets
        target_vals = []
        for t in self.targets:
            target_vals.append([row[f"{t}_kt_{h}"] for h in self.horizons])
        target = torch.tensor(target_vals, dtype=torch.float32)

        # clear sky
        clear_vals = []
        for t in self.targets:
            clear_vals.append([row[f"{t}_clear_{h}"] for h in self.horizons])
        clear_sky = torch.tensor(clear_vals, dtype=torch.float32)

        # elevation
        elevation = torch.tensor(
            [row[f"elevation_{h}"] for h in self.horizons],
            dtype=torch.float32,
        )

        # squeeze for convenience
        if len(self.targets) == 1:
            target = target.squeeze(0)
            clear_sky = clear_sky.squeeze(0)
        if len(self.horizons) == 1:
            target = target.squeeze(-1)
            clear_sky = clear_sky.squeeze(-1)
            elevation = elevation.squeeze()

        timestamp = self.data.index[self.indices[idx]].strftime("%Y%m%d_%H%M%S")

        return {
            "features": features,
            "target": target,
            "clear_sky": clear_sky,
            "elevation": elevation,
            "timestamp": timestamp,
        }


# ==========================================================
# Unit test
# ==========================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing FolsomIntraDayDataset")
    print("=" * 60)

    # 1️⃣ Single target, single horizon
    ds1 = FolsomIntraDayDataset(
        root_dir="/mnt/nfs/yuan/Folsom",
        split="test",
        target="ghi",
        horizon="30min",
    )
    s1 = ds1[0]
    print("\n[TEST 1] Single target / single horizon")
    print("features:", s1["features"].shape)
    print("target:", s1["target"].shape, s1["target"])
    print("clear_sky:", s1["clear_sky"].shape)
    print("elevation:", s1["elevation"])
    print("timestamp:", s1["timestamp"])

    # 2️⃣ Single target, multiple horizons
    ds2 = FolsomIntraDayDataset(
        root_dir="/mnt/nfs/yuan/Folsom",
        split="test",
        target="ghi",
        horizon=["30min", "60min", "90min"],
    )
    s2 = ds2[0]
    print("\n[TEST 2] Single target / multiple horizons")
    print("target shape:", s2["target"].shape)
    print("clear_sky shape:", s2["clear_sky"].shape)
    print("elevation shape:", s2["elevation"].shape)

    # 3️⃣ Multiple targets, single horizon
    ds3 = FolsomIntraDayDataset(
        root_dir="/mnt/nfs/yuan/Folsom",
        split="test",
        target=["ghi", "dni"],
        horizon="30min",
    )
    s3 = ds3[0]
    print("\n[TEST 3] Multiple targets / single horizon")
    print("target shape:", s3["target"].shape)
    print("clear_sky shape:", s3["clear_sky"].shape)

    # 4️⃣ Multiple targets, multiple horizons
    ds4 = FolsomIntraDayDataset(
        root_dir="/mnt/nfs/yuan/Folsom",
        split="test",
        target=["ghi", "dni"],
        horizon=["30min", "60min"],
    )
    s4 = ds4[0]
    print("\n[TEST 4] Multiple targets / multiple horizons")
    print("target shape:", s4["target"].shape)
    print("clear_sky shape:", s4["clear_sky"].shape)
    print("elevation shape:", s4["elevation"].shape)

    print("\n✅ Intra-day dataloader unit tests passed")
