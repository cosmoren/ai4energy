from pathlib import Path
from torch.utils.data import Dataset
from typing import Literal
from datetime import datetime, timedelta
import pandas as pd
import random
import torch

class FolsomIntradayDataset(Dataset):
    """
    PyTorch Dataset for loading Folsom intra-day irradiance features and intra-day targets.
    
    Training set: 2014 and 2015
    Test set: 2016
    """
    
    def __init__(
        self,
        root_dir: str = "/mnt/nfs/yuan/Folsom",
        split: Literal["train", "test"] = "train",
        sample_num: int = 100000
    ):
        """
        Initialize the Folsom intra-day dataset.
        
        Args:
            root_dir: Root directory containing year folders (2014, 2015, 2016)
            split: "train" for 2014+2015, "test" for 2016
        """
        self.root_dir = Path(root_dir)
        self.split = split
        
        # Determine which years to load based on split
        if split == "train":
            year_filter = [2014, 2015]
        elif split == "test":
            year_filter = [2016]
        else:
            raise ValueError(f"split must be 'train' or 'test', got {split}")
        
        # Load irradiance features CSV (intra-day)
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
        
        # Load target CSV (intra-day)
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
        
        # Load satellite CSV
        satellite_csv_path = self.root_dir / "Folsom_satellite.csv"
        if satellite_csv_path.exists():
            print(f"Loading satellite data from {satellite_csv_path}...")
            # Read CSV without header, first column is timestamp
            self.satellite_df = pd.read_csv(satellite_csv_path, header=None)
            # Name the first column as 'timestamp' and the rest as feature columns
            num_cols = len(self.satellite_df.columns)
            self.satellite_df.columns = ['timestamp'] + [f'sat_{i}' for i in range(num_cols - 1)]
            # Parse timestamp column to datetime
            self.satellite_df['timestamp'] = pd.to_datetime(self.satellite_df['timestamp'])
            # Filter by year based on split
            self.satellite_df = self.satellite_df[self.satellite_df['timestamp'].dt.year.isin(year_filter)]
            # Reset index after filtering
            self.satellite_df = self.satellite_df.reset_index(drop=True)
            # Create a mapping from timestamp string to array of satellite features
            self.satellite_df['timestamp_str'] = self.satellite_df['timestamp'].dt.strftime('%Y%m%d_%H%M%S')
            # Exclude "timestamp" and "timestamp_str" from feature columns
            feature_columns = [col for col in self.satellite_df.columns if col not in ('timestamp', 'timestamp_str')]
            self.satellite_dict = {
                row['timestamp_str']: [row[col] for col in feature_columns]
                for _, row in self.satellite_df.iterrows()
            }
            print(f"Loaded {len(self.satellite_df)} satellite records for {split} set (years: {year_filter})")
        else:
            print(f"Warning: Satellite file {satellite_csv_path} not found")
            self.satellite_df = None
            self.satellite_dict = None

        # Select keys based on split type
        if split == "test":
            # For test split, use all available keys (no random sampling)
            self.selected_keys = [key for key in self.irradiance_dict.keys() if key in self.target_dict]
            print(f"Test set: Using all {len(self.selected_keys)} available samples (no random sampling)")
        else:
            # For train split, randomly sample N keys
            N = sample_num
            self.selected_keys = random.sample(list(self.irradiance_dict.keys()), N)
            self.selected_keys = [key for key in self.selected_keys if key in self.target_dict]
            if len(self.selected_keys) != N:
                print(f"Warning: Selected {len(self.selected_keys)} irradiance keys, but only {len(self.target_dict)} target keys exist")
                self.selected_keys = self.selected_keys[:len(self.target_dict)]
    
    def __len__(self):
        return len(self.selected_keys)
    
    def __getitem__(self, idx):
        timestamp_str = self.selected_keys[idx]
        irradiance_data = self.irradiance_dict[timestamp_str]   
        target_data = self.target_dict[timestamp_str]

        # Convert irradiance_data dictionary to tensor
        # Intra-day features have 36 columns: 6 time horizons * 2 types (ghi_kt, dni_kt) * 3 features (B, V, L)
        # CSV order: B(ghi_kt|30min-180min), B(dni_kt|30min-180min), V(ghi_kt|30min-180min), V(dni_kt|30min-180min), L(ghi_kt|30min-180min), L(dni_kt|30min-180min)
        # Reshape to [6, 6] where 6 is time horizons (30min, 60min, 90min, 120min, 150min, 180min) and 6 is (B_ghi, B_dni, V_ghi, V_dni, L_ghi, L_dni)
        time_horizons = ['30min', '60min', '90min', '120min', '150min', '180min']
        feature_types = ['B', 'V', 'L']
        irradiance_types = ['ghi_kt', 'dni_kt']
        
        # Reorganize features: for each time horizon, extract [B_ghi, B_dni, V_ghi, V_dni, L_ghi, L_dni]
        feature_matrix = []
        for horizon in time_horizons:
            horizon_features = []
            for feat_type in feature_types:
                for irr_type in irradiance_types:
                    col_name = f"{feat_type}({irr_type}|{horizon})"
                    val = irradiance_data.get(col_name, 0.0)
                    # Handle NaN values by replacing with 0
                    val = 0.0 if pd.isna(val) else val
                    horizon_features.append(val)
            feature_matrix.append(horizon_features)
        
        irradiance_tensor = torch.tensor(feature_matrix, dtype=torch.float32)  # [6, 6]

        # Generate last 12 timestamp_str (each 15 minutes before the previous one)
        current_dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        last_12_timestamps = []
        last_12_timestamps.append(timestamp_str)
        for i in range(11):
            # Go back 15 minutes for each step (12 * 15 = 180 minutes = 3 hours)
            past_dt = current_dt - timedelta(minutes=15 * (i + 1))
            past_timestamp_str = past_dt.strftime("%Y%m%d_%H%M%S")
            last_12_timestamps.append(past_timestamp_str)
        # Reverse to get chronological order (oldest first)
        last_12_timestamps = last_12_timestamps[::-1]

        satellite_features_list = []
        satellite_features_mask = []
        for timestamp in last_12_timestamps:
            if timestamp in self.satellite_dict:
                satellite_features = torch.tensor(self.satellite_dict[timestamp], dtype=torch.float32)
                satellite_features_mask.append(True)
            else:
                satellite_features = torch.zeros(100, dtype=torch.float32)
                satellite_features_mask.append(False)
            satellite_features_list.append(satellite_features)
        satellite_features_tensor = torch.stack(satellite_features_list)/255  # [timesteps=12, dim=100]
        satellite_features_mask = torch.tensor(satellite_features_mask, dtype=torch.bool)  # [timesteps=12]
                
        # Convert target_data to tensor
        # Intra-day targets have 6 time horizons: 30min, 60min, 90min, 120min, 150min, 180min
        # Extract ghi_kt and dni_kt for each horizon
        target_tensor = torch.tensor([[target_data['ghi_kt_30min'], target_data['ghi_kt_60min'], target_data['ghi_kt_90min'],
                                       target_data['ghi_kt_120min'], target_data['ghi_kt_150min'], target_data['ghi_kt_180min']],
                                      [target_data['dni_kt_30min'], target_data['dni_kt_60min'], target_data['dni_kt_90min'],
                                       target_data['dni_kt_120min'], target_data['dni_kt_150min'], target_data['dni_kt_180min']]],
                                      dtype=torch.float32)  #[Dim, T]

        # Return in standard PyTorch dataloader format (dictionary)
        return {
            'timestamp': timestamp_str,
            'irradiance': irradiance_tensor,
            'target': target_tensor,
            'satellite_features': satellite_features_tensor,
            'satellite_features_mask': satellite_features_mask,
        }


if __name__ == "__main__":
    # Test the dataset
    print("Testing FolsomIntradayDataset...")
    
    # Create train dataset
    print("\n=== Creating train dataset ===")
    train_dataset = FolsomIntradayDataset(
        root_dir="/mnt/nfs/yuan/Folsom",
        split="train",
        sample_num=100
    )
    print(f"Train dataset length: {len(train_dataset)}")
    
    # Create test dataset
    print("\n=== Creating test dataset ===")
    test_dataset = FolsomIntradayDataset(
        root_dir="/mnt/nfs/yuan/Folsom",
        split="test"
    )
    print(f"Test dataset length: {len(test_dataset)}")
    
    # Get a sample from train dataset
    if len(train_dataset) > 0:
        print("\n=== Sample from train dataset ===")
        sample = train_dataset[0]
        print(f"Timestamp: {sample['timestamp']}")
        print(f"Irradiance tensor shape: {sample['irradiance'].shape}")
        print(f"Target tensor shape: {sample['target'].shape}")
        print(f"Irradiance tensor:\n{sample['irradiance']}")
        print(f"Target tensor:\n{sample['target']}")
    
    # Get a sample from test dataset
    if len(test_dataset) > 0:
        print("\n=== Sample from test dataset ===")
        sample = test_dataset[0]
        print(f"Timestamp: {sample['timestamp']}")
        print(f"Irradiance tensor shape: {sample['irradiance'].shape}")
        print(f"Target tensor shape: {sample['target'].shape}")

