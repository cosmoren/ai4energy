from pathlib import Path
from torch.utils.data import Dataset
from typing import Literal, Optional, Union, List
import pandas as pd
import random
import torch
import numpy as np
import pickle
import hashlib
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule


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
        feature_return: Literal["flat", "structured"] = "flat",
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
            feature_return: "flat" returns a single stacked feature vector (backward compatible).
                           "structured" returns a dict with keys: endo, nam_cc, nam.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.feature_return = feature_return
        
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
        
        # Extract features
        if self.feature_return == "flat":
            # Endogenous + NAM cloud cover + NAM irradiance forecasts (backward compatible)
            feature_values = row[self.feature_cols].values.astype(np.float32)
            features = torch.tensor(feature_values, dtype=torch.float32)
        elif self.feature_return == "structured":
            # Structured features for multi-horizon / multi-target models
            endo_values = row[self.feature_cols_endo].values.astype(np.float32)
            endo = torch.tensor(endo_values, dtype=torch.float32)

            cc_cols = [f"nam_cc_{h}" for h in self.horizons]
            nam_cc_values = row[cc_cols].values.astype(np.float32)
            nam_cc = torch.tensor(nam_cc_values, dtype=torch.float32)  # [num_horizons]

            nam_cols = [f"nam_{t}_{h}" for t in self.targets for h in self.horizons]
            nam_values = row[nam_cols].values.astype(np.float32).reshape(len(self.targets), len(self.horizons))
            nam = torch.tensor(nam_values, dtype=torch.float32)  # [num_targets, num_horizons]

            features = {"endo": endo, "nam_cc": nam_cc, "nam": nam}
        else:
            raise ValueError(f"Unknown feature_return: {self.feature_return}")
        
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

        # Actual irradiance for all target-horizon combinations
        actual_values = []
        for t in self.targets:
            actual_row = []
            for h in self.horizons:
                actual_key = f"{t}_{h}"
                actual_value = float(row[actual_key])
                actual_row.append(actual_value)
            actual_values.append(actual_row)
        actual = torch.tensor(actual_values, dtype=torch.float32)

        # NAM irradiance forecast baseline for all target-horizon combinations
        nam_values = []
        for t in self.targets:
            nam_row = []
            for h in self.horizons:
                nam_key = f"nam_{t}_{h}"
                nam_value = float(row[nam_key])
                nam_row.append(nam_value)
            nam_values.append(nam_row)
        nam_irr = torch.tensor(nam_values, dtype=torch.float32)

        # Squeeze actual/nam_irr like target/clear_sky for backward compatibility
        if len(self.targets) == 1 and len(self.horizons) == 1:
            actual = actual.squeeze()
            nam_irr = nam_irr.squeeze()
        elif len(self.targets) == 1:
            actual = actual.squeeze(0)  # [num_horizons]
            nam_irr = nam_irr.squeeze(0)  # [num_horizons]
        elif len(self.horizons) == 1:
            actual = actual.squeeze(1)  # [num_targets]
            nam_irr = nam_irr.squeeze(1)  # [num_targets]
        
        return {
            'features': features,  # Combined endogenous + exogenous features
            'target': target,  # kt values [num_targets, num_horizons] or squeezed
            'clear_sky': clear_sky,  # Clear-sky irradiance [num_targets, num_horizons] or squeezed
            'elevation': elevation,  # Solar elevation [num_horizons] or scalar
            'actual': actual,  # Irradiance [W/m^2] for selected target(s)/horizon(s)
            'nam_irr': nam_irr,  # NAM irradiance baseline [W/m^2] for selected target(s)/horizon(s)
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

    def get_feature_columns(self, feature_set: Literal["all", "endo", "exo", "endo+NAM", "exo+NAM"] = "all") -> List[str]:
        """
        Return the dataframe columns used for a given feature set.

        - all: endogenous + NAM cloud cover + NAM irradiance forecasts (default, matches __getitem__)
        - endo: endogenous only
        - exo: endogenous + NAM cloud cover (nam_cc_{h})
        - endo+NAM: endogenous + NAM irradiance (nam_{t}_{h})
        - exo+NAM: endogenous + NAM cloud cover + NAM irradiance (same as all)
        """
        cc_cols = [f"nam_cc_{h}" for h in self.horizons]
        nam_cols = [f"nam_{t}_{h}" for t in self.targets for h in self.horizons]

        if feature_set == "endo":
            return self.feature_cols_endo
        if feature_set == "exo":
            return list(dict.fromkeys(self.feature_cols_endo + cc_cols))
        if feature_set == "endo+NAM":
            return list(dict.fromkeys(self.feature_cols_endo + nam_cols))
        if feature_set == "exo+NAM" or feature_set == "all":
            return list(dict.fromkeys(self.feature_cols_endo + cc_cols + nam_cols))
        raise ValueError(f"Unknown feature_set: {feature_set}")

    def get_features(self, feature_set: Literal["all", "endo", "exo", "endo+NAM", "exo+NAM"] = "all") -> torch.Tensor:
        """
        Return a stacked feature matrix with shape [N, D] as a torch float tensor.
        The stacking order matches self.indices (the dataset's sampling order).
        """
        df = self.data.iloc[self.indices]
        cols = self.get_feature_columns(feature_set=feature_set)
        return torch.tensor(df[cols].values.astype(np.float32), dtype=torch.float32)

    def get_index(self):
        """Return the pandas index for the sampled rows (in the same order as returned tensors)."""
        return self.data.iloc[self.indices].index

    def _stack_matrix(self, col_names: List[str], shape: tuple) -> torch.Tensor:
        df = self.data.iloc[self.indices]
        x = df[col_names].values.astype(np.float32)
        return torch.tensor(x.reshape(shape), dtype=torch.float32)

    def get_kt(self) -> torch.Tensor:
        """Return kt target tensor with shape [N, num_targets, num_horizons]."""
        cols = [f"{t}_kt_{h}" for t in self.targets for h in self.horizons]
        return self._stack_matrix(cols, (len(self.indices), len(self.targets), len(self.horizons)))

    def get_clear_sky(self) -> torch.Tensor:
        """Return clear-sky irradiance tensor with shape [N, num_targets, num_horizons]."""
        cols = [f"{t}_clear_{h}" for t in self.targets for h in self.horizons]
        return self._stack_matrix(cols, (len(self.indices), len(self.targets), len(self.horizons)))

    def get_actual(self) -> torch.Tensor:
        """Return actual irradiance tensor with shape [N, num_targets, num_horizons]."""
        cols = [f"{t}_{h}" for t in self.targets for h in self.horizons]
        return self._stack_matrix(cols, (len(self.indices), len(self.targets), len(self.horizons)))

    def get_nam(self) -> torch.Tensor:
        """
        Return NAM irradiance forecast tensor with shape [N, num_targets, num_horizons].
        (This is the day-ahead baseline in the official scripts.)
        """
        cols = [f"nam_{t}_{h}" for t in self.targets for h in self.horizons]
        return self._stack_matrix(cols, (len(self.indices), len(self.targets), len(self.horizons)))

    def get_elevation(self) -> torch.Tensor:
        """Return elevation tensor with shape [N, num_horizons]."""
        df = self.data.iloc[self.indices]
        cols = [f"elevation_{h}" for h in self.horizons]
        return torch.tensor(df[cols].values.astype(np.float32), dtype=torch.float32)


def _to_hashable(x):
    if x is None:
        return None
    if isinstance(x, list):
        return tuple(x)
    return x


def get_cache_key(
    root_dir: str,
    split: str,
    sample_num: Optional[int],
    target: Optional[Union[str, List[str]]],
    horizon: Optional[Union[str, List[str]]],
) -> str:
    """Generate a cache key based on dataset parameters."""
    key_string = f"{root_dir}_{split}_{sample_num}_{_to_hashable(target)}_{_to_hashable(horizon)}"
    return hashlib.md5(key_string.encode()).hexdigest()


class FolsomDayAheadDataModule(LightningDataModule):
    """
    Thin PyTorch Lightning DataModule wrapper for Folsom day-ahead dataset.
    """

    def __init__(
        self,
        root_dir: str = "/mnt/nfs/yuan/Folsom",
        target: Optional[Union[str, List[str]]] = "ghi",
        horizon: Optional[Union[str, List[str]]] = "26h",
        train_sample_num: Optional[int] = None,
        val_sample_num: Optional[int] = None,
        test_sample_num: Optional[int] = None,
        batch_size: int = 64,
        num_workers: int = 8,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        pin_memory: bool = True,
        drop_last: bool = True,
        persistent_workers: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["cache_dir"])

        self.root_dir = root_dir
        self.target = target
        self.horizon = horizon
        self.train_sample_num = train_sample_num
        self.val_sample_num = val_sample_num
        self.test_sample_num = test_sample_num
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_cache = use_cache
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.persistent_workers = persistent_workers and num_workers > 0

        if cache_dir is None:
            self.cache_dir = Path(__file__).parent.parent / ".cache" / "datasets"
        else:
            self.cache_dir = Path(cache_dir)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = self._load_or_create_dataset(
                split="train",
                sample_num=self.train_sample_num,
            )
            # Default: use test split as validation (consistent with previous intra-hour wrapper)
            self.val_dataset = self._load_or_create_dataset(
                split="test",
                sample_num=self.val_sample_num,
            )

        if stage == "test" or stage is None:
            self.test_dataset = self._load_or_create_dataset(
                split="test",
                sample_num=self.test_sample_num,
            )

    def _load_or_create_dataset(self, split: str, sample_num: Optional[int]) -> "FolsomDayAheadDataset":
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        cache_key = get_cache_key(
            root_dir=self.root_dir,
            split=split,
            sample_num=sample_num,
            target=self.target,
            horizon=self.horizon,
        )
        cache_file = self.cache_dir / f"dataset_{split}_{cache_key}.pkl"

        if self.use_cache and cache_file.exists():
            with open(cache_file, "rb") as f:
                dataset = pickle.load(f)
            return dataset

        dataset = FolsomDayAheadDataset(
            root_dir=self.root_dir,
            split=split,
            target=self.target,
            horizon=self.horizon,
            sample_num=sample_num,
        )

        if self.use_cache:
            with open(cache_file, "wb") as f:
                pickle.dump(dataset, f)

        return dataset

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is None. Call setup('fit') first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is None. Call setup('fit') first.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("test_dataset is None. Call setup('test') first.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.persistent_workers,
        )


if __name__ == "__main__":
    split = "train" # can be "train" or "test"
    target = None # can be "ghi" or "dni" or None for all targets
    horizon = None # can be "26h", "27h", "28h", "29h", "30h", "31h", "32h", "33h", "34h", "35h", "36h", "37h", "38h", "39h" or None for all horizons

    root_dir = Path("/mnt/nfs/yuan/Folsom")
    required = [
        root_dir / "Irradiance_features_day-ahead.csv",
        root_dir / "NAM_nearest_node_day-ahead.csv",
        root_dir / "Target_day-ahead.csv",
    ]
    if not all(p.exists() for p in required):
        print("SKIP: missing required day-ahead CSVs:")
        for p in required:
            if not p.exists():
                print(" -", p)
        raise SystemExit(0)

    ds = FolsomDayAheadDataset(
        root_dir=str(root_dir),
        split=split,
        target=target,
        horizon=horizon,
        sample_num=None,
        feature_return="structured",
    )
    print("len:", len(ds))
    sample = ds[0]
    print("keys:", sorted(sample.keys()))
    if isinstance(sample["features"], dict):
        print("features (structured):", {k: tuple(v.shape) for k, v in sample["features"].items()})
    else:
        print("features:", tuple(sample["features"].shape))
    print("target:", tuple(sample["target"].shape) if hasattr(sample["target"], "shape") else type(sample["target"]))
    print("clear_sky:", tuple(sample["clear_sky"].shape) if hasattr(sample["clear_sky"], "shape") else type(sample["clear_sky"]))
    print("elevation:", tuple(sample["elevation"].shape) if hasattr(sample["elevation"], "shape") else type(sample["elevation"]))
    print("timestamp:", sample["timestamp"])