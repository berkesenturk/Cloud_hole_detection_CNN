"""
PyTorch Dataset for Gold (Training-Ready) Data

This dataset loads pre-processed data from gold Zarr and applies augmentation.
All other preprocessing (filtering, resizing, normalization) is already done.

Usage:
    from gold_dataset import GoldCloudHoleDataset

    # Training (with augmentation)
    train_dataset = GoldCloudHoleDataset(
        gold_zarr_path='data/gold/datasets/v1.0_baseline/train/data.zarr',
        augment=True
    )

    # Validation/Test (no augmentation)
    val_dataset = GoldCloudHoleDataset(
        gold_zarr_path='data/gold/datasets/v1.0_baseline/validation/data.zarr',
        augment=False
    )
"""

import torch
import xarray as xr
from torch.utils.data import Dataset
from torchvision import transforms
import random


class CloudHoleDataset(Dataset):
    """
    PyTorch Dataset for pre-processed gold data.

    Data is already:
    - Filtered by years
    - Split into train/val/test
    - Single timestep per sample
    - Resized to 224x224
    - Normalized

    Only augmentation is applied here (on-the-fly, training only).
    """

    def __init__(
            self,
            gold_zarr_path,
            pretrained=False,
            model=None,
            augment=False
    ):
        """
        Initialize dataset.

        Args:
            gold_zarr_path: Path to gold Zarr file
                (e.g., 'data/gold/datasets/v1.0_baseline/train/data.zarr')
            augment: Whether to augmentate data (True for training only)
        """
        # Load pre-processed Zarr data
        self.dataset = xr.open_zarr(gold_zarr_path)
        self.augment = augment
        self.n_samples = len(self.dataset.sample)
        self.pretrained = pretrained
        self.model = model

        # Define augmentation pipelines (only if augment=True)
        if self.augment:
            self.augmentation_pipelines = [
                transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(p=1.0),
                    ]
                ),
                transforms.Compose(
                    [
                        transforms.RandomVerticalFlip(p=1.0),
                    ]
                ),
                transforms.Compose(
                    [
                        transforms.RandomRotation(degrees=5),
                    ]
                ),
            ]

        print(f"✓ Loaded GoldCloudHoleDataset")
        print(f"  Path: {gold_zarr_path}")
        print(f"  Samples: {self.n_samples}")
        print(f"  Augmentation: {'ON' if augment else 'OFF'}")

        # Print normalization stats if available
        if hasattr(self.dataset, "attrs"):
            if "mean" in self.dataset.attrs:
                print(
                    f"  Normalization: mean={self.dataset.attrs['mean']:.4f}, "
                    f"std={self.dataset.attrs.get('std', 'N/A')}"
                )

    def __len__(self):
        """Return number of samples"""
        return self.n_samples

    def __getitem__(self, idx):
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            images: Tensor of shape (C, H, W)
            label: Tensor with binary label (0 or 1)
        """
        # Load pre-processed images (already normalized)
        images = torch.from_numpy(self.dataset.images[idx].values).float()

        if self.pretrained and self.model == "resnet18":
            # ResNet18 expects 3-channel input,
            # so we repeat the single channel 3 times
            images = images.repeat(3, 1, 1)

        # Load label
        label = torch.tensor(int(self.dataset.labels[idx].values), dtype=torch.long)

        # Apply augmentation only during training
        if self.augment:
            # Randomly choose one augmentation pipeline
            pipeline = random.choice(self.augmentation_pipelines)
            images = pipeline(images)

        return images, label

    def get_normalization_stats(self):
        """
        Get normalization statistics used for this dataset.

        Returns:
            dict: Dictionary with 'mean', 'std', 'min', 'max'
        """
        return {
            "mean": float(self.dataset.attrs.get("mean", 0)),
            "std": float(self.dataset.attrs.get("std", 1)),
            "min": float(self.dataset.attrs.get("min", 0)),
            "max": float(self.dataset.attrs.get("max", 1)),
        }

    def get_sample_date(self, idx):
        """
        Get the date for a specific sample.

        Args:
            idx: Sample index

        Returns:
            str: Date string
        """
        return str(self.dataset.dates[idx].values)

    def get_class_distribution(self):
        """
        Get class distribution in this dataset.

        Returns:
            dict: Dictionary with counts for each class
        """
        labels = self.dataset.labels.values
        return {
            "cloud_hole": int((labels == 1).sum()),
            "non_cloud_hole": int((labels == 0).sum()),
            "total": len(labels),
        }
