import os
import numpy as np
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
        image_size: int = 10,
        image_his_len = 3,
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
        self.image_his_len = image_his_len
        
        # Determine which years to load based on split
        if split == "train":
            years = ["2014", "2015"]
            year_filter = [2014, 2015]
        elif split == "test":
            years = ["2016"]
            year_filter = [2016]
        else:
            raise ValueError(f"split must be 'train' or 'test', got {split}")
        
        # load sky image features
        sky_feat_csv_path = self.root_dir / "Sky_image_features_intra-hour.csv"
        if sky_feat_csv_path.exists():
            print(f"Loading sky features from {sky_feat_csv_path}...")
            self.sky_df = pd.read_csv(sky_feat_csv_path)
            # Parse timestamp column to datetime
            self.sky_df['timestamp'] = pd.to_datetime(self.sky_df['timestamp'])
            # Filter by year based on split
            self.sky_df = self.sky_df[self.sky_df['timestamp'].dt.year.isin(year_filter)]
            # Reset index after filtering
            self.sky_df = self.sky_df.reset_index(drop=True)
            # Create a mapping from timestamp (same format as Target_intra-day.csv key) to feature dict
            self.sky_df['timestamp_str'] = self.sky_df['timestamp'].dt.strftime('%Y%m%d_%H%M%S')
            # Exclude "timestamp" and "timestamp_str" columns from feature dicts
            feature_columns = [col for col in self.sky_df.columns if col not in ('timestamp', 'timestamp_str')]
            self.sky_dict = {
                row['timestamp_str']: {col: row[col] for col in feature_columns}
                for _, row in self.sky_df.iterrows()
            }
            print(f"Loaded {len(self.sky_df)} sky feature records for {split} set (years: {year_filter})")
        else:
            print(f"Warning: sky features file {sky_feat_csv_path} not found")
            self.sky_df = None
            self.sky_dict = None

        # Load satellite features
        sat_feat_csv_path = self.root_dir / "Sat_image_features_intra-day.csv"
        if sat_feat_csv_path.exists():
            print(f"Loading sat image features from {sat_feat_csv_path}...")
            sat_feat_df = pd.read_csv(
                sat_feat_csv_path,
                parse_dates=True,
                index_col=0,   # index is timestamp
            )
            # filter by year
            sat_feat_df = sat_feat_df[sat_feat_df.index.year.isin(year_filter)]
            # make timestamp_str for lookup consistency
            sat_feat_df["timestamp_str"] = sat_feat_df.index.strftime("%Y%m%d_%H%M%S")
            feat_cols = sat_feat_df.columns.drop("timestamp_str")
            self.sat_feat_dict = {
                row["timestamp_str"]: row[feat_cols].values.astype("float32")
                for _, row in sat_feat_df.iterrows()
            }
            self.sat_feat_dim = len(feat_cols)
            print(f"Loaded {len(self.sat_feat_dict)} satellite feature records "
                f"(dim={self.sat_feat_dim})")
        else:
            print(f"Warning: {sat_feat_csv_path} not found")
            self.sat_feat_dict = None
            self.sat_feat_dim = 0

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
        available_keys = sorted(set(self.irradiance_dict.keys()) & set(self.target_dict.keys()))
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

                # ---------- satellite (his_len frames required) ----------
                t_issue = pd.Timestamp(datetime.strptime(ts, "%Y%m%d_%H%M%S"))
                pos = sat_times.searchsorted(
                    t_issue.to_datetime64(),
                    side="right"
                ) - 1

                # need his_len frames
                if pos < self.image_his_len - 1:
                    continue

                valid_keys.append(ts)

            print(
                f"[Train] filtered {len(self.selected_keys)} -> {len(valid_keys)} samples "
                f"(require full his_len frame satellite window)"
            )
            self.selected_keys = valid_keys        
    
    def __len__(self):
        return len(self.selected_keys)
    
    def __getitem__(self, idx):
        # time stamp and target
        timestamp_str = self.selected_keys[idx]
        irradiance_data = self.irradiance_dict[timestamp_str]
        target_data = self.target_dict[timestamp_str]

        t_issue = pd.Timestamp(
            datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        )
        
        # make satellite image window（his_len frames）
        sat_times = self.sat_df["timestamp"].values  # np.datetime64 array

        # find last position where <= t_issue holds（this is the current frame）
        pos = sat_times.searchsorted(t_issue.to_datetime64(), side="right") - 1
        #print("timestamp_str ", timestamp_str, " sat_times at pos ", sat_times[pos])
        # init window: [his_len, 1, 10, 10]，all NaN
        sat_window = torch.full(
            (self.image_his_len, 1, 10, 10),
            float("nan"),
            dtype=torch.float32,
        )

        # If there is no sat img at all before current time
        if pos < 0:
            # keep the whole window NaN
            pass
        else:
            # fill from (pos-(his_len-1)) to pos
            for i in range(self.image_his_len):
                sat_idx = pos - (self.image_his_len - 1 - i)   # i=his_len -> pos（current）
                if sat_idx < 0:
                    continue
                if sat_idx >= len(self.sat_df):
                    continue

                patch = self.sat_df.iloc[sat_idx, 1:].values.astype("float32")
                patch = torch.from_numpy(patch).view(1, 10, 10)

                sat_window[i] = patch
        # encode relative times
        time_delta = torch.empty(self.image_his_len, dtype=torch.float32)
        for i in range(self.image_his_len):
            sat_idx = pos - (self.image_his_len - 1 - i)
            if sat_idx < 0:
                time_delta[i] = float("nan")
            else:
                dt_min = (t_issue.to_datetime64() - sat_times[sat_idx]) / np.timedelta64(1, "m")
                time_delta[i] = dt_min
        time_delta = time_delta.view(self.image_his_len, 1, 1, 1).expand(-1, -1, 10, 10) # [his_len, 1, 10, 10]
        time_delta = 255.0 * torch.exp(-(time_delta) / 90.0) # exp decay, 0->255, inf->0
        sat_window = torch.cat([sat_window, time_delta], dim = 1) # [his_len, 2, 10, 10]
        # Irradiance tensor
        irradiance_tensor = torch.tensor(
            [irradiance_data[k] for k in irradiance_data.keys()],
            dtype=torch.float32,
        ).reshape(6, 6)

        irradiance_tensor = torch.fliplr(irradiance_tensor)  # [6, 6]

        # Target tensor
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

        # clear sky kt
        clear_tensor = torch.tensor(
            [
                [
                    self.irradiance_dict[timestamp_str]["B(ghi_kt|30min)"],
                    self.irradiance_dict[timestamp_str]["B(ghi_kt|60min)"],
                    self.irradiance_dict[timestamp_str]["B(ghi_kt|90min)"],
                    self.irradiance_dict[timestamp_str]["B(ghi_kt|120min)"],
                    self.irradiance_dict[timestamp_str]["B(ghi_kt|150min)"],
                    self.irradiance_dict[timestamp_str]["B(ghi_kt|180min)"],
                ],
                [
                    self.irradiance_dict[timestamp_str]["B(dni_kt|30min)"],
                    self.irradiance_dict[timestamp_str]["B(dni_kt|60min)"],
                    self.irradiance_dict[timestamp_str]["B(dni_kt|90min)"],
                    self.irradiance_dict[timestamp_str]["B(dni_kt|120min)"],
                    self.irradiance_dict[timestamp_str]["B(dni_kt|150min)"],
                    self.irradiance_dict[timestamp_str]["B(dni_kt|180min)"],
                ],
            ],
            dtype=torch.float32,
        )
        
        '''
        self.irradiance_dict[timestamp_str]["B(ghi_kt|30min)"],
                    self.irradiance_dict[timestamp_str]["V(ghi_kt|30min)"],

                    self.irradiance_dict[timestamp_str]["B(ghi_kt|60min)"],
                    self.irradiance_dict[timestamp_str]["V(ghi_kt|60min)"],

                    self.irradiance_dict[timestamp_str]["B(ghi_kt|120min)"],
                    self.irradiance_dict[timestamp_str]["V(ghi_kt|120min)"],

                    self.irradiance_dict[timestamp_str]["B(ghi_kt|180min)"],
                    self.irradiance_dict[timestamp_str]["L(ghi_kt|180min)"],           
        '''

        # xgb_input_tensor to replicate paper
        xgb_input_tensor = torch.tensor(
            [
                [
                    [self.irradiance_dict[timestamp_str]["B(ghi_kt|30min)"], self.irradiance_dict[timestamp_str]["V(ghi_kt|30min)"], self.sky_dict[timestamp_str]['ENT(R)'], self.sky_dict[timestamp_str]['AVG(G)'], self.sky_dict[timestamp_str]['ENT(RB)']],
                    [self.irradiance_dict[timestamp_str]["B(ghi_kt|60min)"], self.irradiance_dict[timestamp_str]["V(ghi_kt|60min)"], self.sky_dict[timestamp_str]['AVG(R)'], self.sky_dict[timestamp_str]['ENT(G)'], self.sky_dict[timestamp_str]['ENT(RB)']],
                    [0, 0, 0, 0, 0],
                    [self.irradiance_dict[timestamp_str]["B(ghi_kt|120min)"], self.irradiance_dict[timestamp_str]["V(ghi_kt|120min)"], self.sky_dict[timestamp_str]['AVG(G)'], self.sky_dict[timestamp_str]['ENT(G)'], self.sky_dict[timestamp_str]['ENT(RB)']],
                    [0, 0, 0, 0, 0],
                    [self.irradiance_dict[timestamp_str]["B(ghi_kt|180min)"], self.irradiance_dict[timestamp_str]["L(ghi_kt|180min)"], self.sky_dict[timestamp_str]['AVG(G)'], self.sky_dict[timestamp_str]['ENT(B)'], self.sky_dict[timestamp_str]['ENT(RB)']],
                ],
                [
                    [self.irradiance_dict[timestamp_str]["B(dni_kt|30min)"], self.irradiance_dict[timestamp_str]["V(dni_kt|30min)"], self.sky_dict[timestamp_str]['ENT(R)'], self.sky_dict[timestamp_str]['AVG(G)'], self.sky_dict[timestamp_str]['ENT(RB)']],
                    [self.irradiance_dict[timestamp_str]["B(dni_kt|60min)"], self.irradiance_dict[timestamp_str]["V(dni_kt|60min)"], self.sky_dict[timestamp_str]['ENT(G)'], self.sky_dict[timestamp_str]['AVG(G)'], self.sky_dict[timestamp_str]['ENT(RB)']],
                    [0, 0, 0, 0, 0],
                    [self.irradiance_dict[timestamp_str]["V(dni_kt|120min)"], self.sky_dict[timestamp_str]['ENT(R)'], self.sky_dict[timestamp_str]['AVG(G)'], self.sky_dict[timestamp_str]['AVG(B)'], self.sky_dict[timestamp_str]['ENT(RB)']],
                    [0, 0, 0, 0, 0],
                    [self.irradiance_dict[timestamp_str]["V(dni_kt|180min)"], self.irradiance_dict[timestamp_str]["L(dni_kt|180min)"], self.sky_dict[timestamp_str]['AVG(R)'], self.sky_dict[timestamp_str]['ENT(R)'], self.sky_dict[timestamp_str]['ENT(RB)']],
                ]
            ]
        ) # [2, 6, 5] 2 modes(ghi,dni), 6 horizons, 5 inputs

        # return
        return {
            "timestamp": timestamp_str,
            "irradiance": irradiance_tensor,   # [6, 6]
            "images": sat_window,            # [his_len, 1, 10, 10]
            "clear_kt": clear_tensor,        # [2, 6]
            "xgb_input": xgb_input_tensor, # [2, 6, 5]
            "target": target_tensor,            # [2, 6]
        }



