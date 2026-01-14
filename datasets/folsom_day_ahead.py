from pathlib import Path
from torch.utils.data import Dataset
from typing import Literal, Optional, Union, List
import pandas as pd
import random
import torch
import numpy as np


class FolsomDayAheadDataset(Dataset):
    """
    PyTorch Dataset for loading Folsom day-ahead forecasting data.
    
    This dataset matches the structure used in official/Forecast_day-ahead.py:
    - Training set: years <= 2015 (2014 and 2015)
    - Test set: year == 2016
    
    Day-ahead horizons: 26h, 27h, 28h, 29h, 30h, 31h, 32h, 33h,
                        34h, 35h, 36h, 37h, 38h, 39h (14 horizons)
    
    Supports multiple targets and/or multiple horizons:
    - target: "ghi", "dni", ["ghi"], ["dni"], ["ghi", "dni"], or None (all targets)
    - horizon: single string, list of strings, or None (all 14 horizons)
    
    The dataset returns:
    - features: combined endogenous (irradiance features filtered by targets) 
                and exogenous (nam_cc_{horizon}, nam_{target}_{horizon}) features
    - target: kt values with shape [num_targets, num_horizons]
    - clear_sky: clear-sky irradiance values with shape [num_targets, num_horizons]
    - elevation: solar elevation angles with shape [num_horizons]
    """
    
    # All available horizons for day-ahead forecasting
    ALL_HORIZONS = ["26h", "27h", "28h", "29h", "30h", "31h", "32h", "33h",
                     "34h", "35h", "36h", "37h", "38h", "39h"]
    # All available targets
    ALL_TARGETS = ["ghi", "dni"]
    
    def __init__(
        self,
        root_dir: str = "/mnt/nfs/yuan/Folsom",
        split: Literal["train", "test"] = "train",
        target: Optional[Union[str, List[str]]] = "ghi",
        horizon: Optional[Union[str, List[str]]] = "26h",
        sample_num: Optional[int] = None,
    ):
        """
        Initialize the Folsom day-ahead dataset.
        
        Args:
            root_dir: Root directory containing CSV files
            split: "train" for years <= 2015, "test" for year == 2016
            target: Target variable(s). Can be:
                - Single string: "ghi" or "dni"
                - List of strings: ["ghi"], ["dni"], or ["ghi", "dni"]
                - None: uses all targets ["ghi", "dni"]
            horizon: Forecast horizon(s). Can be:
                - Single string: "26h", "27h", etc.
                - List of strings: ["26h", "27h"], etc.
                - None: uses all 14 horizons
            sample_num: Number of samples to randomly sample for training (None = use all)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        
        # Normalize target(s)
        if target is None:
            self.targets = self.ALL_TARGETS.copy()
        elif isinstance(target, str):
            self.targets = [target]
        elif isinstance(target, list):
            self.targets = target
        else:
            raise ValueError(f"target must be str, list of str, or None, got {type(target)}")
        
        # Validate targets
        for t in self.targets:
            if t not in self.ALL_TARGETS:
                raise ValueError(f"target must be 'ghi' or 'dni', got {t}")
        self.targets = sorted(set(self.targets))  # Remove duplicates and sort
        
        # Normalize horizon(s)
        if horizon is None:
            self.horizons = self.ALL_HORIZONS.copy()
        elif isinstance(horizon, str):
            self.horizons = [horizon]
        elif isinstance(horizon, list):
            self.horizons = horizon
        else:
            raise ValueError(f"horizon must be str, list of str, or None, got {type(horizon)}")
        
        # Validate horizons
        for h in self.horizons:
            if not h.endswith("h") or not h[:-1].isdigit():
                raise ValueError(f"horizon must be in format like '26h', '27h', etc., got {h}")
            if h not in self.ALL_HORIZONS:
                raise ValueError(f"horizon {h} not in allowed horizons: {self.ALL_HORIZONS}")
        self.horizons = sorted(set(self.horizons), key=lambda x: int(x[:-1]))  # Remove duplicates and sort
        
        # Store single target/horizon for backward compatibility
        self.target = self.targets[0] if len(self.targets) == 1 else None
        self.horizon = self.horizons[0] if len(self.horizons) == 1 else None
        
        # Load irradiance features CSV (endogenous features)
        # CSV format: timestamp as index (first column), parse_dates=True
        irradiance_csv_path = self.root_dir / "Irradiance_features_day-ahead.csv"
        if not irradiance_csv_path.exists():
            raise FileNotFoundError(f"Irradiance features file not found: {irradiance_csv_path}")
        
        print(f"Loading irradiance features from {irradiance_csv_path}...")
        inpEndo = pd.read_csv(irradiance_csv_path, delimiter=",", parse_dates=True, index_col=0)
        
        # Load NAM weather model features CSV (exogenous features)
        nam_csv_path = self.root_dir / "NAM_nearest_node_day-ahead.csv"
        if not nam_csv_path.exists():
            raise FileNotFoundError(f"NAM features file not found: {nam_csv_path}")
        
        print(f"Loading NAM features from {nam_csv_path}...")
        inpExo = pd.read_csv(nam_csv_path, delimiter=",", parse_dates=True, index_col=0)
        
        # Load target CSV
        target_csv_path = self.root_dir / "Target_day-ahead.csv"
        if not target_csv_path.exists():
            raise FileNotFoundError(f"Target file not found: {target_csv_path}")
        
        print(f"Loading target data from {target_csv_path}...")
        tar = pd.read_csv(target_csv_path, delimiter=",", parse_dates=True, index_col=0)
        
        # Filter by year and join all dataframes (matching official code exactly)
        if split == "train":
            # Train: years <= 2015 (matching official code: inpEndo.index.year <= 2015)
            train = inpEndo[inpEndo.index.year <= 2015]
            train = train.join(inpExo[inpEndo.index.year <= 2015], how="inner")
            train = train.join(tar[tar.index.year <= 2015], how="inner")
            data = train
        elif split == "test":
            # Test: year == 2016 (matching official code: inpEndo.index.year == 2016)
            test = inpEndo[inpEndo.index.year == 2016]
            test = test.join(inpExo[inpEndo.index.year == 2016], how="inner")
            test = test.join(tar[tar.index.year == 2016], how="inner")
            data = test
        else:
            raise ValueError(f"split must be 'train' or 'test', got {split}")
        
        # Define columns needed for all target-horizon combinations
        cols = []
        for t in self.targets:
            for h in self.horizons:
                cols.extend([
                    f"{t}_{h}",  # actual
                    f"{t}_kt_{h}",  # clear-sky index (target)
                    f"{t}_clear_{h}",  # clear-sky model
                ])
        
        # Add elevation for all horizons (elevation is horizon-specific, not target-specific)
        for h in self.horizons:
            cols.append(f"elevation_{h}")
        
        # Get feature columns (matching official code pattern)
        # Endogenous features: union of features matching any target regex
        feature_cols_endo = []
        for t in self.targets:
            target_features = inpEndo.filter(regex=t).columns.tolist()
            feature_cols_endo.extend(target_features)
        feature_cols_endo = sorted(set(feature_cols_endo))  # Remove duplicates and sort
        
        # Exogenous features: nam_cc_{horizon} for each horizon, and nam_{target}_{horizon} for each target-horizon pair
        feature_cols = feature_cols_endo.copy()
        for h in self.horizons:
            feature_cols.append(f"nam_cc_{h}")
        for t in self.targets:
            for h in self.horizons:
                feature_cols.append(f"nam_{t}_{h}")
        
        # Select only the columns we need and drop any rows with NaN (matching official code)
        required_cols = cols + feature_cols
        # Remove duplicates while preserving order
        required_cols = list(dict.fromkeys(required_cols))
        data = data[required_cols].dropna(how="any")
        
        # Store data
        self.data = data
        self.feature_cols = feature_cols
        self.feature_cols_endo = feature_cols_endo
        
        # Get indices (samples)
        self.indices = list(range(len(data)))
        
        # Random sampling for training if requested
        if split == "train" and sample_num is not None:
            N = min(sample_num, len(self.indices))
            self.indices = random.sample(self.indices, N)
            print(f"Train set: Randomly sampled {len(self.indices)} samples from {len(data)} available")
        else:
            print(f"{split.capitalize()} set: Using all {len(self.indices)} available samples")
        
        print(f"Dataset configuration:")
        print(f"  Targets: {self.targets}")
        print(f"  Horizons: {self.horizons}")
        print(f"  Endogenous features: {len(self.feature_cols_endo)}")
        print(f"  Total features: {len(self.feature_cols)}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Returns:
            Dictionary with:
            - 'features': torch.Tensor of shape [num_features] - combined endogenous and exogenous features
            - 'target': torch.Tensor of shape [num_targets, num_horizons] - kt values for all target-horizon combinations
            - 'clear_sky': torch.Tensor of shape [num_targets, num_horizons] - clear-sky irradiance values
            - 'elevation': torch.Tensor of shape [num_horizons] - solar elevation angles
            - 'timestamp': str - timestamp string for reference
            
            If single target and single horizon, target and clear_sky are scalars (0-d tensors).
            If single target and multiple horizons, target and clear_sky are 1-d tensors [num_horizons].
            If multiple targets and single horizon, target and clear_sky are 1-d tensors [num_targets].
            If multiple targets and multiple horizons, target and clear_sky are 2-d tensors [num_targets, num_horizons].
        """
        actual_idx = self.indices[idx]
        row = self.data.iloc[actual_idx]
        
        # Extract features (matching official code order)
        feature_values = row[self.feature_cols].values.astype(np.float32)
        features = torch.tensor(feature_values, dtype=torch.float32)
        
        # Extract targets (kt values) for all target-horizon combinations
        target_values = []
        for t in self.targets:
            target_row = []
            for h in self.horizons:
                target_key = f"{t}_kt_{h}"
                target_value = float(row[target_key])
                target_row.append(target_value)
            target_values.append(target_row)
        target = torch.tensor(target_values, dtype=torch.float32)
        
        # Squeeze dimensions for backward compatibility (single target/horizon)
        if len(self.targets) == 1 and len(self.horizons) == 1:
            target = target.squeeze()
        elif len(self.targets) == 1:
            target = target.squeeze(0)  # [num_horizons]
        elif len(self.horizons) == 1:
            target = target.squeeze(1)  # [num_targets]
        
        # Extract clear-sky values for all target-horizon combinations
        clear_sky_values = []
        for t in self.targets:
            clear_sky_row = []
            for h in self.horizons:
                clear_sky_key = f"{t}_clear_{h}"
                clear_sky_value = float(row[clear_sky_key])
                clear_sky_row.append(clear_sky_value)
            clear_sky_values.append(clear_sky_row)
        clear_sky = torch.tensor(clear_sky_values, dtype=torch.float32)
        
        # Squeeze dimensions for backward compatibility
        if len(self.targets) == 1 and len(self.horizons) == 1:
            clear_sky = clear_sky.squeeze()
        elif len(self.targets) == 1:
            clear_sky = clear_sky.squeeze(0)  # [num_horizons]
        elif len(self.horizons) == 1:
            clear_sky = clear_sky.squeeze(1)  # [num_targets]
        
        # Extract elevation for all horizons (elevation is horizon-specific, not target-specific)
        elevation_values = []
        for h in self.horizons:
            elevation_key = f"elevation_{h}"
            elevation_value = float(row[elevation_key])
            elevation_values.append(elevation_value)
        elevation = torch.tensor(elevation_values, dtype=torch.float32)
        
        # Squeeze for single horizon
        if len(self.horizons) == 1:
            elevation = elevation.squeeze()
        
        # Get timestamp for reference
        timestamp = self.data.index[actual_idx]
        timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S')
        
        return {
            'features': features,  # Combined endogenous + exogenous features
            'target': target,  # kt values [num_targets, num_horizons] or squeezed
            'clear_sky': clear_sky,  # Clear-sky irradiance [num_targets, num_horizons] or squeezed
            'elevation': elevation,  # Solar elevation [num_horizons] or scalar
            'timestamp': timestamp_str,  # Timestamp string
        }
    
    def get_endo_features_only(self, idx):
        """
        Get only endogenous features (without exogenous features).
        This matches the 'endo' model variant in the official code.
        
        Args:
            idx: Sample index
            
        Returns:
            torch.Tensor of shape [num_endo_features] - only endogenous features
        """
        actual_idx = self.indices[idx]
        row = self.data.iloc[actual_idx]
        
        # Extract only endogenous features
        feature_values = row[self.feature_cols_endo].values.astype(np.float32)
        return torch.tensor(feature_values, dtype=torch.float32)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Test the dataset
    print("Testing FolsomDayAheadDataset...")
    print("=" * 60)
    
    # Test 1: Single target, single horizon (backward compatibility)
    print("\n" + "=" * 60)
    print("Test 1: Single target, single horizon")
    print("=" * 60)
    test_dataset1 = FolsomDayAheadDataset(
        root_dir="/mnt/nfs/yuan/Folsom",
        split="test",
        target="ghi",
        horizon="26h"
    )
    sample1 = test_dataset1[0]
    print(f"\nSample structure:")
    print(f"  Features shape: {sample1['features'].shape}")
    print(f"  Target shape: {sample1['target'].shape if sample1['target'].dim() > 0 else 'scalar'}, value: {sample1['target'].item() if sample1['target'].dim() == 0 else sample1['target']}")
    print(f"  Clear-sky shape: {sample1['clear_sky'].shape if sample1['clear_sky'].dim() > 0 else 'scalar'}, value: {sample1['clear_sky'].item() if sample1['clear_sky'].dim() == 0 else sample1['clear_sky']}")
    print(f"  Elevation shape: {sample1['elevation'].shape if sample1['elevation'].dim() > 0 else 'scalar'}, value: {sample1['elevation'].item() if sample1['elevation'].dim() == 0 else sample1['elevation']}")
    print(f"  Timestamp: {sample1['timestamp']}")
    
    # Test 2: Single target, multiple horizons
    print("\n" + "=" * 60)
    print("Test 2: Single target, multiple horizons")
    print("=" * 60)
    test_dataset2 = FolsomDayAheadDataset(
        root_dir="/mnt/nfs/yuan/Folsom",
        split="test",
        target="ghi",
        horizon=["26h", "27h", "28h"]
    )
    sample2 = test_dataset2[0]
    print(f"\nSample structure:")
    print(f"  Features shape: {sample2['features'].shape}")
    print(f"  Target shape: {sample2['target'].shape}")
    print(f"  Clear-sky shape: {sample2['clear_sky'].shape}")
    print(f"  Elevation shape: {sample2['elevation'].shape}")
    print(f"  Timestamp: {sample2['timestamp']}")
    
    # Test 3: Multiple targets, single horizon
    print("\n" + "=" * 60)
    print("Test 3: Multiple targets, single horizon")
    print("=" * 60)
    test_dataset3 = FolsomDayAheadDataset(
        root_dir="/mnt/nfs/yuan/Folsom",
        split="test",
        target=["ghi", "dni"],
        horizon="26h"
    )
    sample3 = test_dataset3[0]
    print(f"\nSample structure:")
    print(f"  Features shape: {sample3['features'].shape}")
    print(f"  Target shape: {sample3['target'].shape}")
    print(f"  Clear-sky shape: {sample3['clear_sky'].shape}")
    print(f"  Elevation shape: {sample3['elevation'].shape if sample3['elevation'].dim() > 0 else 'scalar'}")
    print(f"  Timestamp: {sample3['timestamp']}")
    
    # Test 4: Multiple targets, multiple horizons
    print("\n" + "=" * 60)
    print("Test 4: Multiple targets, multiple horizons")
    print("=" * 60)
    test_dataset4 = FolsomDayAheadDataset(
        root_dir="/mnt/nfs/yuan/Folsom",
        split="test",
        target=["ghi", "dni"],
        horizon=["26h", "27h"]
    )
    sample4 = test_dataset4[0]
    print(f"\nSample structure:")
    print(f"  Features shape: {sample4['features'].shape}")
    print(f"  Target shape: {sample4['target'].shape}")
    print(f"  Clear-sky shape: {sample4['clear_sky'].shape}")
    print(f"  Elevation shape: {sample4['elevation'].shape}")
    print(f"  Timestamp: {sample4['timestamp']}")
    
    # Test 5: All targets and all horizons
    print("\n" + "=" * 60)
    print("Test 5: All targets and all horizons (None)")
    print("=" * 60)
    test_dataset5 = FolsomDayAheadDataset(
        root_dir="/mnt/nfs/yuan/Folsom",
        split="test",
        target=None,  # All targets
        horizon=None  # All horizons
    )
    sample5 = test_dataset5[0]
    print(f"\nSample structure:")
    print(f"  Features shape: {sample5['features'].shape}")
    print(f"  Target shape: {sample5['target'].shape}")
    print(f"  Clear-sky shape: {sample5['clear_sky'].shape}")
    print(f"  Elevation shape: {sample5['elevation'].shape}")
    print(f"  Timestamp: {sample5['timestamp']}")