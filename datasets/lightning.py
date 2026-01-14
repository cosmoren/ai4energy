"""
PyTorch Lightning DataModule wrapper for Folsom dataset.

This module provides a Lightning DataModule that wraps the FolsomDataset,
simplifying data loading for PyTorch Lightning training workflows.
"""

import pickle
import hashlib
from pathlib import Path
from typing import Optional
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from .folsom import FolsomDataset


def get_cache_key(root_dir: str, split: str, sample_num: Optional[int], image_size: int) -> str:
    """Generate a cache key based on dataset parameters."""
    key_string = f"{root_dir}_{split}_{sample_num}_{image_size}"
    return hashlib.md5(key_string.encode()).hexdigest()


class FolsomDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for Folsom dataset.
    
    This DataModule handles:
    - Dataset initialization for train/val/test splits
    - Caching of datasets to disk
    - DataLoader creation with proper configuration
    - Multi-GPU support via Lightning's distributed data loading
    """
    
    def __init__(
        self,
        root_dir: str = "/mnt/nfs/yuan/Folsom",
        train_sample_num: int = 100000,
        val_sample_num: int = 100000,
        test_sample_num: Optional[int] = None,
        batch_size: int = 4,
        num_workers: int = 8,
        image_size: int = 224,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        pin_memory: bool = True,
        drop_last: bool = True,
        persistent_workers: bool = True,
    ):
        """
        Initialize the Folsom DataModule.
        
        Args:
            root_dir: Root directory containing year folders (2014, 2015, 2016)
            train_sample_num: Number of training samples to use
            val_sample_num: Number of validation samples to use
            test_sample_num: Number of test samples to use (None = use all available)
            batch_size: Batch size per GPU
            num_workers: Number of data loading workers per GPU
            image_size: Size of input images (default: 448)
            cache_dir: Directory for caching datasets (default: .cache/datasets)
            use_cache: Whether to use dataset caching (default: True)
            pin_memory: Whether to pin memory in DataLoader (default: True)
            drop_last: Whether to drop last incomplete batch in training (default: True)
            persistent_workers: Whether to keep workers alive between epochs (default: True)
        """
        super().__init__()
        self.save_hyperparameters(ignore=['cache_dir'])
        
        self.root_dir = root_dir
        self.train_sample_num = train_sample_num
        self.val_sample_num = val_sample_num
        self.test_sample_num = test_sample_num
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.use_cache = use_cache
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.persistent_workers = persistent_workers and num_workers > 0
        
        # Set default cache directory
        if cache_dir is None:
            self.cache_dir = Path(__file__).parent.parent / ".cache" / "datasets"
        else:
            self.cache_dir = Path(cache_dir)
        
        # Datasets will be initialized in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets for the specified stage.
        
        Args:
            stage: 'fit' for train/val, 'test' for test, or None for all
        """
        if stage == "fit" or stage is None:
            self.train_dataset = self._load_or_create_dataset(
                split="train",
                sample_num=self.train_sample_num
            )
            self.val_dataset = self._load_or_create_dataset(
                split="test",  # Using test split as validation
                sample_num=self.val_sample_num
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = self._load_or_create_dataset(
                split="test",
                sample_num=self.test_sample_num
            )
    
    def _load_or_create_dataset(
        self,
        split: str,
        sample_num: Optional[int]
    ) -> FolsomDataset:
        """
        Load dataset from cache or create new one.
        
        Args:
            split: Dataset split ("train" or "test")
            sample_num: Number of samples to use (None = use all available)
        
        Returns:
            FolsomDataset instance
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_key = get_cache_key(self.root_dir, split, sample_num, self.image_size)
        cache_file = self.cache_dir / f"dataset_{split}_{cache_key}.pkl"
        
        if self.use_cache and cache_file.exists():
            print(f"Loading {split} dataset from cache: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    dataset = pickle.load(f)
                print(f"Successfully loaded cached {split} dataset ({len(dataset)} samples)")
                return dataset
            except Exception as e:
                print(f"Warning: Failed to load cache ({e}), creating new dataset...")
        
        # Create new dataset
        print(f"Creating {split} dataset (this may take a while)...")
        dataset = FolsomDataset(
            root_dir=self.root_dir,
            split=split,
            sample_num=sample_num,
            image_size=self.image_size
        )
        
        # Save to cache
        if self.use_cache:
            print(f"Saving {split} dataset to cache: {cache_file}")
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(dataset, f)
                print(f"Successfully cached {split} dataset")
            except Exception as e:
                print(f"Warning: Failed to save cache ({e})")
        
        return dataset
    
    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader."""
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is None. Call setup('fit') first.")
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader."""
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is None. Call setup('fit') first.")
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.persistent_workers
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader."""
        if self.test_dataset is None:
            raise RuntimeError("test_dataset is None. Call setup('test') first.")
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.persistent_workers
        )
