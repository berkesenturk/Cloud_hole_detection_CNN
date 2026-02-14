"""
Prepare Training-Ready Gold Dataset from Silver Zarr Data

This script creates fully preprocessed, training-ready data:
- Filters by years for train/val/test splits
- Uses single timesteps (one image per sample)
- Resizes to 224×224
- Normalizes (using train statistics)
- Saves as Zarr in gold container

Augmentation is NOT applied here - it's done on-the-fly in PyTorch Dataset

Usage:
    python prepare_gold.py
"""

import os
import json
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
SILVER_ZARR_PATH = os.getenv('silver_data_path') + '/seviri/hrv_lr2004_2019.zarr'
LABELS_PATH = os.getenv('processed_data_path') + '/labels_revised.csv'
GOLD_BASE_PATH = os.getenv('gold_data_path') + '/seviri/'

# Version
VERSION = 'v1.0_baseline'

# Split configuration
TRAIN_YEARS = [2005, 2007, 2013]
VAL_YEARS = [2015]
TEST_YEARS = [2016]

# Preprocessing parameters
IMAGE_SIZE = (224, 224)
STANDARD_NORMALIZE = False  # False = min-max, True = standard


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_silver_data(zarr_path):
    """Load silver Zarr dataset"""
    print(f"\n{'='*80}")
    print(f"Loading silver Zarr from: {zarr_path}")
    print(f"{'='*80}\n")
    
    dataset = xr.open_zarr(zarr_path)
    print(f"✓ Loaded dataset")
    print(f"  Shape: {dataset.hrv.shape}")
    print(f"  Time range: {dataset.time.min().values} to {dataset.time.max().values}")
    
    return dataset


def load_labels(labels_path):
    """Load and clean labels CSV"""
    print(f"\nLoading labels from: {labels_path}")
    
    labels_df = pd.read_csv(labels_path, index_col=0, parse_dates=True)
    
    # Clean labels
    labels_df = labels_df.dropna(subset=["label"])
    labels_df = labels_df[~labels_df["label"].str.contains("problem", na=False)]
    
    print(f"✓ Loaded {len(labels_df)} valid labels")
    print(f"  Cloud holes: {(labels_df['label'] == 'cloud_hole').sum()}")
    print(f"  Non-cloud holes: {(labels_df['label'] != 'cloud_hole').sum()}")
    
    return labels_df


def filter_by_years(dataset, labels_df, years, split_name):
    """Filter dataset and labels by specified years"""
    print(f"\n{'='*80}")
    print(f"Processing {split_name.upper()} split - Years: {years}")
    print(f"{'='*80}\n")
    
    # Filter labels by years
    labels_filtered = labels_df[
        pd.DatetimeIndex(labels_df.index).year.isin(years)
    ].sort_index()
    
    print(f"✓ Filtered labels to {len(labels_filtered)} samples for years {years}")
    
    return labels_filtered


def create_samples(dataset, labels_df):
    """Create samples from single timesteps"""
    samples_data = []
    samples_labels = []
    samples_dates = []
    
    print(f"\nCreating samples from single timesteps...")
    
    for date in tqdm(labels_df.index, desc="Processing"):
        try:
            # Extract data for this single timestamp
            sample_data = dataset.hrv.sel(time=date)
            
            # Check for NaN values
            if sample_data.isnull().any():
                continue
            
            # Get label
            label_row = labels_df.loc[date]
            if label_row['label'] == 'cloud_hole':
                label = 1
            elif label_row['label'] == 'non_cloud_hole':
                label = 0
            else:
                continue
            # Add channel dimension if needed (make it 3D: C, H, W)
            if sample_data.ndim == 2:
                # If grayscale (H, W), add channel dimension
                sample_data = sample_data.expand_dims('channel', axis=0)
            
            samples_data.append(sample_data.values)
            samples_labels.append(label)
            samples_dates.append(str(date))
            
        except Exception as e:
            continue
    
    print(f"✓ Created {len(samples_data)} valid samples")
    
    return samples_data, samples_labels, samples_dates


def resize_samples(samples, target_size):
    """Resize samples using bilinear interpolation"""
    print(f"\nResizing samples to {target_size}...")
    
    resized_samples = []
    
    for sample in tqdm(samples, desc="Resizing"):
        # sample shape: (C, H, W) or (H, W)
        
        # Ensure we have 3D array (C, H, W)
        if sample.ndim == 2:
            # Grayscale (H, W) -> add channel dimension
            sample = sample[np.newaxis, :, :]
        
        # Get dimensions
        n_channels = sample.shape[0]
        
        # Create DataArray for interpolation
        da = xr.DataArray(
            sample,
            dims=['channel', 'y', 'x'],
            coords={
                'channel': np.arange(n_channels),
                'y': np.arange(sample.shape[1]),
                'x': np.arange(sample.shape[2])
            }
        )
        
        # New coordinates for target size
        y_new = np.linspace(0, sample.shape[1]-1, target_size[0])
        x_new = np.linspace(0, sample.shape[2]-1, target_size[1])
        
        # Interpolate
        resized = da.interp(y=y_new, x=x_new, method='linear')
        
        resized_samples.append(resized.values)
    
    print(f"✓ Resized {len(resized_samples)} samples")
    
    return resized_samples


def calculate_normalization_stats(samples):
    """Calculate normalization statistics from samples"""
    print(f"\nCalculating normalization statistics...")
    
    all_data = np.stack(samples, axis=0)
    
    stats = {
        'mean': float(np.mean(all_data)),
        'std': float(np.std(all_data)),
        'min': float(np.min(all_data)),
        'max': float(np.max(all_data))
    }
    
    print(f"✓ Statistics:")
    print(f"  Mean: {stats['mean']:.4f}")
    print(f"  Std:  {stats['std']:.4f}")
    print(f"  Min:  {stats['min']:.4f}")
    print(f"  Max:  {stats['max']:.4f}")
    
    return stats


def normalize_samples(samples, stats, standard_normalize):
    """Normalize samples using provided statistics"""
    print(f"\nNormalizing samples...")
    print(f"  Method: {'Standard (z-score)' if standard_normalize else 'Min-Max [0,1]'}")
    
    normalized = []
    
    for sample in samples:
        if standard_normalize:
            # Standard normalization: (x - mean) / std
            norm_sample = (sample - stats['mean']) / stats['std']
        else:
            # Min-max normalization: (x - min) / (max - min)
            norm_sample = (sample - stats['min']) / (stats['max'] - stats['min'])
        
        normalized.append(norm_sample)
    
    print(f"✓ Normalized {len(normalized)} samples")
    
    return normalized


def save_split_to_zarr(samples, labels, dates, split_name, output_dir, stats=None, image_size=(224, 224)):
    """Save split data to Zarr format"""
    print(f"\n{'='*80}")
    print(f"Saving {split_name.upper()} split to Zarr")
    print(f"{'='*80}\n")
    
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    
    # Stack samples into array
    data_array = np.stack(samples, axis=0)  # Shape: (N, C, H, W)
    labels_array = np.array(labels, dtype=np.int32)
    dates_array = np.array(dates, dtype='object')
    
    n_samples = len(samples)
    n_channels = data_array.shape[1]
    
    print(f"  Samples: {n_samples}")
    print(f"  Shape: {data_array.shape}")
    print(f"  Channels: {n_channels}")
    print(f"  Cloud holes: {np.sum(labels_array == 1)}")
    print(f"  Non-cloud holes: {np.sum(labels_array == 0)}")
    
    # Create xarray Dataset
    dataset = xr.Dataset(
        {
            'images': (
                ['sample', 'channel', 'y', 'x'], 
                data_array,
                {'description': 'Preprocessed satellite images (resized and normalized)'}
            ),
            'labels': (
                ['sample'], 
                labels_array,
                {'description': 'Binary labels: 1=cloud_hole, 0=non_cloud_hole'}
            ),
            'dates': (
                ['sample'], 
                dates_array,
                {'description': 'Date for each sample'}
            )
        },
        coords={
            'sample': np.arange(n_samples),
            'channel': np.arange(n_channels),
            'y': np.arange(image_size[0]),
            'x': np.arange(image_size[1])
        },
        attrs={
            'version': VERSION,
            'split': split_name,
            'created_at': datetime.now().isoformat(),
            'image_size': list(image_size),
            'n_channels': n_channels,
            'preprocessing': 'resized and normalized, single timestep per sample',
        }
    )
    
    # Add normalization stats to attributes
    if stats:
        dataset.attrs.update(stats)
    
    # Save to Zarr with compression
    zarr_path = split_dir / 'data.zarr'
    
    if zarr_path.exists():
        import shutil
        shutil.rmtree(zarr_path)
    
    # Import numcodecs for compression
    import numcodecs
    
    # Encoding with compression
    encoding = {
        'images': {
            'compressor': numcodecs.Zlib(level=4),
            'chunks': (100, n_channels, image_size[0], image_size[1])
        },
        'labels': {
            'compressor': numcodecs.Zlib(level=4)
        },
    }
    
    dataset.to_zarr(zarr_path, mode='w', encoding=encoding)
    
    print(f"\n✓ Saved Zarr to: {zarr_path}")
    
    # Save metadata
    metadata_dir = split_dir / 'metadata'
    metadata_dir.mkdir(exist_ok=True)
    
    metadata = {
        'split': split_name,
        'num_samples': int(n_samples),
        'class_distribution': {
            'cloud_hole': int(np.sum(labels_array == 1)),
            'non_cloud_hole': int(np.sum(labels_array == 0))
        },
        'image_shape': [n_channels, image_size[0], image_size[1]],
        'zarr_path': str(zarr_path),
        'created_at': datetime.now().isoformat()
    }
    
    with open(metadata_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save normalization stats (train only)
    if stats:
        with open(metadata_dir / 'normalization_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Saved normalization stats to: {metadata_dir / 'normalization_stats.json'}")
    
    print(f"✓ Saved metadata to: {metadata_dir}")
    
    return metadata


def process_split(dataset, labels_df, years, split_name, output_dir, norm_stats=None):
    """Process a complete split"""
    
    # Filter by years
    labels_filtered = filter_by_years(dataset, labels_df, years, split_name)
    
    # Create samples
    samples_data, samples_labels, samples_dates = create_samples(
        dataset, labels_filtered
    )
    
    if len(samples_data) == 0:
        raise ValueError(f"No valid samples created for {split_name} split!")
    
    # Resize samples
    resized_samples = resize_samples(samples_data, IMAGE_SIZE)
    
    # Calculate or use normalization stats
    if norm_stats is None and split_name == 'train':
        norm_stats = calculate_normalization_stats(resized_samples)
    elif norm_stats is None:
        raise ValueError(f"Normalization stats required for {split_name} split")
    
    # Normalize samples
    normalized_samples = normalize_samples(resized_samples, norm_stats, STANDARD_NORMALIZE)
    
    # Save to Zarr
    metadata = save_split_to_zarr(
        normalized_samples,
        samples_labels,
        samples_dates,
        split_name,
        output_dir,
        stats=norm_stats if split_name == 'train' else None,
        image_size=IMAGE_SIZE
    )
    
    return metadata, norm_stats


def create_version_summary(output_dir, all_metadata):
    """Create overall version summary"""
    
    summary = {
        'version': VERSION,
        'created_at': datetime.now().isoformat(),
        'preprocessing': {
            'train_years': TRAIN_YEARS,
            'val_years': VAL_YEARS,
            'test_years': TEST_YEARS,
            'image_size': list(IMAGE_SIZE),
            'standard_normalize': STANDARD_NORMALIZE,
            'augmentation_applied': False
        },
        'splits': all_metadata,
        'total_samples': sum(m['num_samples'] for m in all_metadata.values())
    }
    
    summary_path = output_dir / 'version_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Saved version summary to: {summary_path}")
    
    return summary


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution"""
    
    print(f"\n{'='*80}")
    print(f"GOLD DATASET PREPARATION")
    print(f"Version: {VERSION}")
    print(f"{'='*80}\n")
    
    # Create output directory
    output_dir = Path(GOLD_BASE_PATH) / VERSION
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    dataset = load_silver_data(SILVER_ZARR_PATH)
    labels_df = load_labels(LABELS_PATH)
    
    all_metadata = {}
    
    # Process train split first (to calculate normalization stats)
    print(f"\n{'#'*80}")
    print(f"# TRAIN SPLIT")
    print(f"{'#'*80}")
    train_metadata, norm_stats = process_split(
        dataset, labels_df, TRAIN_YEARS, 'train', output_dir
    )
    all_metadata['train'] = train_metadata
    
    # Process validation split
    print(f"\n{'#'*80}")
    print(f"# VALIDATION SPLIT")
    print(f"{'#'*80}")
    val_metadata, _ = process_split(
        dataset, labels_df, VAL_YEARS, 'validation', output_dir, norm_stats
    )
    all_metadata['validation'] = val_metadata
    
    # Process test split
    print(f"\n{'#'*80}")
    print(f"# TEST SPLIT")
    print(f"{'#'*80}")
    test_metadata, _ = process_split(
        dataset, labels_df, TEST_YEARS, 'test', output_dir, norm_stats
    )
    all_metadata['test'] = test_metadata
    
    # Create version summary
    summary = create_version_summary(output_dir, all_metadata)
    
    # Print final summary
    print(f"\n{'='*80}")
    print(f"✓ GOLD DATASET PREPARATION COMPLETE")
    print(f"{'='*80}\n")
    print(f"Version: {VERSION}")
    print(f"Output: {output_dir}\n")
    print(f"Summary:")
    print(f"  Train:      {train_metadata['num_samples']:4d} samples ({train_metadata['class_distribution']['cloud_hole']} cloud holes)")
    print(f"  Validation: {val_metadata['num_samples']:4d} samples ({val_metadata['class_distribution']['cloud_hole']} cloud holes)")
    print(f"  Test:       {test_metadata['num_samples']:4d} samples ({test_metadata['class_distribution']['cloud_hole']} cloud holes)")
    print(f"  Total:      {summary['total_samples']:4d} samples\n")
    print(f"Data is training-ready (single timestep, resized & normalized)")
    print(f"Augmentation should be applied in PyTorch Dataset\n")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()