import os
from pathlib import Path
from torch.utils.data import Dataset
from typing import Literal, Optional
import pvlib
from datetime import datetime
import time
import pandas as pd
import random
from PIL import Image
import torch
from torchvision import transforms

class FolsomIntraDayDataset(Dataset):
    """
    PyTorch Dataset for loading Folsom sky images.
    
    Training set: 2014 and 2015
    Test set: 2016
    """
    
    def __init__(
        self,
        root_dir: str = "/mnt/nfs/yuan/Folsom",
        split: Literal["train", "test"] = "train",
        image_extensions: tuple = (".jpg", ".jpeg", ".png"),
        sample_num: Optional[int] = None,
        image_size: int = 10
    ):
        """
        Initialize the Folsom dataset.
        
        Args:
            root_dir: Root directory containing year folders (2014, 2015, 2016)
            split: "train" for 2014+2015, "test" for 2016
            image_extensions: Tuple of valid image file extensions
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_extensions = image_extensions
        self.image_paths = []
        self.image_size = image_size
        
        # Determine which years to load based on split
        if split == "train":
            years = ["2014", "2015"]
            year_filter = [2014, 2015]
        elif split == "test":
            years = ["2016"]
            year_filter = [2016]
        else:
            raise ValueError(f"split must be 'train' or 'test', got {split}")
        
        # Load irradiance features CSV
        irradiance_csv_path = self.root_dir / "Irradiance_features_intra-day.csv"
        if irradiance_csv_path.exists():
            print(f"Loading irradiance features from {irradiance_csv_path}...")
            self.irradiance_df = pd.read_csv(irradiance_csv_path)
            # Parse timestamp column to datetime
            self.irradiance_df['timestamp'] = pd.to_datetime(self.irradiance_df['timestamp'])
            # Filter by year based on split
            self.irradiance_df = self.irradiance_df[self.irradiance_df['timestamp'].dt.year.isin(year_filter)]
            # Reset index after filtering
            self.irradiance_df = self.irradiance_df.reset_index(drop=True)
            # Create a mapping from timestamp (same format as Target_intra-day.csv key) to feature dict
            self.irradiance_df['timestamp_str'] = self.irradiance_df['timestamp'].dt.strftime('%Y%m%d_%H%M%S')
            # Exclude "timestamp" and "timestamp_str" columns from feature dicts
            feature_columns = [col for col in self.irradiance_df.columns if col not in ('timestamp', 'timestamp_str')]
            self.irradiance_dict = {
                row['timestamp_str']: {col: row[col] for col in feature_columns}
                for _, row in self.irradiance_df.iterrows()
            }
            print(f"Loaded {len(self.irradiance_df)} irradiance feature records for {split} set (years: {year_filter})")
        else:
            print(f"Warning: Irradiance features file {irradiance_csv_path} not found")
            self.irradiance_df = None
            self.irradiance_dict = None
        
        # Load target CSV
        target_csv_path = self.root_dir / "Target_intra-day.csv"
        if target_csv_path.exists():
            print(f"Loading target data from {target_csv_path}...")
            self.target_df = pd.read_csv(target_csv_path)
            # Parse timestamp column to datetime
            self.target_df['timestamp'] = pd.to_datetime(self.target_df['timestamp'])
            # Filter by year based on split
            self.target_df = self.target_df[self.target_df['timestamp'].dt.year.isin(year_filter)]
            # Reset index after filtering
            self.target_df = self.target_df.reset_index(drop=True)
            # Create a mapping from timestamp string to dict of all columns except timestamp and timestamp_str
            self.target_df['timestamp_str'] = self.target_df['timestamp'].dt.strftime('%Y%m%d_%H%M%S')
            # Exclude "timestamp" and "timestamp_str" from value dicts
            feature_columns = [col for col in self.target_df.columns if col not in ('timestamp', 'timestamp_str')]
            self.target_dict = {
                row['timestamp_str']: {col: row[col] for col in feature_columns}
                for _, row in self.target_df.iterrows()
            }
            print(f"Loaded {len(self.target_df)} target records for {split} set (years: {year_filter})")
        else:
            print(f"Warning: Target file {target_csv_path} not found")
            self.target_df = None
            self.target_dict = None
   
        # Collect images
        sat_df = pd.read_csv(self.root_dir / "Folsom_satellite.csv", header=None)
        sat_df.columns = ["timestamp"] + [f"p{i}" for i in range(100)]
        sat_df["timestamp"] = pd.to_datetime(sat_df["timestamp"])
        sat_df = sat_df.sort_values("timestamp").reset_index(drop=True)
        self.sat_df = sat_df
   
        # Get available keys (intersection of irradiance and target dicts)
        available_keys = list(set(self.irradiance_dict.keys()) & set(self.target_dict.keys()))
        # Select keys: if sample_num is None, use all available; otherwise sample N
        if sample_num is None:
            self.selected_keys = available_keys
            print(f"Using all {len(self.selected_keys)} available samples")
        else:
            if len(available_keys) < sample_num:
                print(f"Warning: Only {len(available_keys)} keys available, requested {sample_num}. Using all available.")
                self.selected_keys = available_keys
            else:
                self.selected_keys = random.sample(available_keys, sample_num)

        # validity check, training split only
        if split == "train":
            valid_keys = []

            sat_times = self.sat_df["timestamp"].values

            for ts in self.selected_keys:
                # ---------- irradiance ----------
                irr = self.irradiance_dict[ts]
                if any(pd.isna(v) for v in irr.values()):
                    continue

                # ---------- target ----------
                tgt = self.target_dict[ts]
                if any(pd.isna(v) for v in tgt.values()):
                    continue

                # ---------- satellite (12 frames required) ----------
                t_issue = pd.Timestamp(datetime.strptime(ts, "%Y%m%d_%H%M%S"))
                pos = sat_times.searchsorted(
                    t_issue.to_datetime64(),
                    side="right"
                ) - 1

                # need 12 frames
                if pos < 11:
                    continue

                valid_keys.append(ts)

            print(
                f"[Train] filtered {len(self.selected_keys)} -> {len(valid_keys)} samples "
                f"(require full 12-frame satellite window)"
            )
            self.selected_keys = valid_keys
        
    
    def __len__(self):
        return len(self.selected_keys)
    
    def __getitem__(self, idx):
        # 1. time stamp and target
        timestamp_str = self.selected_keys[idx]
        irradiance_data = self.irradiance_dict[timestamp_str]
        target_data = self.target_dict[timestamp_str]

        t_issue = pd.Timestamp(
            datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        )
        
        # 2. make satellite image window（12 frames）
        sat_times = self.sat_df["timestamp"].values  # np.datetime64 array

        # find last position where <= t_issue holds（this is the current frame）
        pos = sat_times.searchsorted(t_issue.to_datetime64(), side="right") - 1
        #print("timestamp_str ", timestamp_str, " sat_times at pos ", sat_times[pos])
        # init window: [12, 1, 10, 10]，all NaN
        sat_window = torch.full(
            (12, 1, 10, 10),
            float("nan"),
            dtype=torch.float32,
        )

        # If there is no sat img at all before current time
        if pos < 0:
            # keep the whole window NaN
            pass
        else:
            # fill from (pos-11) to pos
            for i in range(12):
                sat_idx = pos - (11 - i)   # i=11 -> pos（current）
                if sat_idx < 0:
                    continue
                if sat_idx >= len(self.sat_df):
                    continue

                patch = self.sat_df.iloc[sat_idx, 1:].values.astype("float32")
                patch = torch.from_numpy(patch).view(1, 10, 10)

                sat_window[i] = patch

        # 3. Irradiance tensor
        irradiance_tensor = torch.tensor(
            [irradiance_data[k] for k in irradiance_data.keys()],
            dtype=torch.float32,
        ).reshape(6, 6)

        irradiance_tensor = torch.fliplr(irradiance_tensor)  # [6, 6]

        # 4. Target tensor
        target_tensor = torch.tensor(
            [
                [
                    target_data["ghi_kt_30min"],
                    target_data["ghi_kt_60min"],
                    target_data["ghi_kt_90min"],
                    target_data["ghi_kt_120min"],
                    target_data["ghi_kt_150min"],
                    target_data["ghi_kt_180min"],
                ],
                [
                    target_data["dni_kt_30min"],
                    target_data["dni_kt_60min"],
                    target_data["dni_kt_90min"],
                    target_data["dni_kt_120min"],
                    target_data["dni_kt_150min"],
                    target_data["dni_kt_180min"],
                ],
            ],
            dtype=torch.float32,
        )  # [2, 6]

        # 5. return
        return {
            "timestamp": timestamp_str,
            "irradiance": irradiance_tensor,   # [6, 6]
            "images": sat_window,            # [12, 1, 10, 10]
            "target": target_tensor,            # [2, 6]
        }



