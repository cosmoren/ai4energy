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

class FolsomDataset(Dataset):
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
        sample_num: int = 100000,
        image_size: int = 224
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
        irradiance_csv_path = self.root_dir / "Irradiance_features_intra-hour.csv"
        if irradiance_csv_path.exists():
            print(f"Loading irradiance features from {irradiance_csv_path}...")
            self.irradiance_df = pd.read_csv(irradiance_csv_path)
            # Parse timestamp column to datetime
            self.irradiance_df['timestamp'] = pd.to_datetime(self.irradiance_df['timestamp'])
            # Filter by year based on split
            self.irradiance_df = self.irradiance_df[self.irradiance_df['timestamp'].dt.year.isin(year_filter)]
            # Reset index after filtering
            self.irradiance_df = self.irradiance_df.reset_index(drop=True)
            # Create a mapping from timestamp (same format as Target_intra-hour.csv key) to feature dict
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
        target_csv_path = self.root_dir / "Target_intra-hour.csv"
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
   
        # Collect all image paths from the specified years
        for year in years:
            year_dir = self.root_dir / year
            if not year_dir.exists():
                print(f"Warning: Year directory {year_dir} does not exist, skipping...")
                continue
            
            # Iterate through months (01-12)
            for month in range(1, 13):
                month_dir = year_dir / f"{month:02d}"
                if not month_dir.exists():
                    continue
                
                # Iterate through days (01-31)
                for day in range(1, 32):
                    day_dir = month_dir / f"{day:02d}"
                    if not day_dir.exists():
                        continue
                    
                    # Collect all image files in the day directory
                    for file_path in day_dir.iterdir():
                        if file_path.is_file() and file_path.suffix.lower() in self.image_extensions:
                            self.image_paths.append(file_path)
        
        # Sort image paths chronologically
        # The filename format is YYYYMMDD_HHMMSS.jpg, so we can sort by filename
        image_info_list = []
        for file_path in self.image_paths:
            # Parse filename, expecting format 'yyyymmdd_hhmmss.*'
            name = file_path.stem  # removes suffix
            try:
                dt = datetime.strptime(name, "%Y%m%d_%H%M%S")
                unix_timestamp = int(time.mktime(dt.timetuple()))
            except ValueError:
                print(f"Warning: Filename {file_path} does not match expected format yyyymmdd_hhmmss.*")
                continue
            jd = pvlib.spa.julian_day(unix_timestamp)
            image_info_list.append({'JD': jd, 'path': file_path})

        # Sort by Julian Day
        image_info_list.sort(key=lambda x: x['JD'])
        self.image_paths = image_info_list
        
        print(f"Loaded {len(self.image_paths)} images for {split} set")

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

        dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        unix_timestamp = int(time.mktime(dt.timetuple()))
        timestamp_jd = pvlib.spa.julian_day(unix_timestamp)

        timestamp_jd0 = timestamp_jd - 1/48
        
        # Filter image paths whose JD is between timestamp_jd0 and timestamp_jd
        image_paths_in_window = [
            img_info['path'] for img_info in self.image_paths
            if timestamp_jd0 <= img_info['JD'] <= timestamp_jd
        ]
        
        # Read images one by one and resize to 448x448
        images = []
        for img_path in image_paths_in_window:
            try:
                img = Image.open(img_path)
                # img = img.resize((self.image_size, self.image_size))
                images.append(img)
            except Exception as e:
                print(f"Warning: Failed to load image {img_path}: {e}")
                continue
        
        # Convert images from list to Nx3ximage_sizeximage_size tensor
        if len(images) > 0:
            to_tensor = transforms.ToTensor()
            image_tensors = [to_tensor(img) for img in images]
            images_tensor = torch.stack(image_tensors)  # Shape: Nx3ximage_sizeximage_size
        else:
            images_tensor = torch.empty((0, 3, self.image_size, self.image_size))
        
        # Ensure images_tensor has shape [30, 3, image_size, image_size]
        target_num_images = 30
        current_num = images_tensor.shape[0]
        
        if current_num > target_num_images:
            images_tensor = images_tensor[current_num-target_num_images:]
        elif current_num < target_num_images:
            # If fewer than 30 images, pad with zeros at the beginning
            num_to_pad = target_num_images - current_num
            zero_padding = torch.zeros((num_to_pad, 3, self.image_size, self.image_size), dtype=images_tensor.dtype)
            images_tensor = torch.cat([zero_padding, images_tensor], dim=0)

        # Convert irradiance_data and target_data dictionaries to tensors
        irradiance_tensor = torch.fliplr(torch.tensor([irradiance_data[key] for key in irradiance_data.keys()], dtype=torch.float32).reshape(6, 6)) # [Channel, T]     
        
        target_tensor = torch.tensor([[target_data['ghi_kt_5min'], target_data['ghi_kt_10min'], target_data['ghi_kt_15min'],
                                       target_data['ghi_kt_20min'], target_data['ghi_kt_25min'], target_data['ghi_kt_30min']],
                                      [target_data['dni_kt_5min'], target_data['dni_kt_10min'], target_data['dni_kt_15min'],
                                       target_data['dni_kt_20min'], target_data['dni_kt_25min'], target_data['dni_kt_30min']]],
                                      dtype=torch.float32)  #[Dim, T]

        # Return in standard PyTorch dataloader format (dictionary)
        return {
            'timestamp': timestamp_str,
            'irradiance': irradiance_tensor,
            'target': target_tensor,
            'images': images_tensor
        }

