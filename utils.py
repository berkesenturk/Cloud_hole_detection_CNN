import torch
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import random
import os
from collections import Counter
import math

cmap = plt.get_cmap('Greys_r')

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        
        inputs = F.softmax(inputs.logits, dim=1) 

        targets = F.one_hot(targets, num_classes=self.num_classes).float() 

        log_p = torch.log(inputs + 1e-8)  

        ce_loss = -targets * log_p  

        p_t = torch.sum(inputs * targets, dim=1, keepdim=True)  

        focal_weight = (1 - p_t) ** self.gamma  

        loss = focal_weight * ce_loss 
        loss = self.alpha * loss  

        return loss.sum(dim=1).mean()


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def plot_subplots(filtered_data, title="Augmented Images", figsize=(10, 10), idx_channel = None):
    """
    Plot a grid of subplots for a given list of image data.

    Parameters:
    - filtered_data: List of tuples (date, image_data).
    - title: Title for the entire plot.
    - figsize: Tuple specifying figure size.
    """
    num_images = len(filtered_data)
    if num_images == 0:
        print("No images to plot.")
        return
    
    # Calculate grid size
    cols = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()  # Flatten axes for easy iteration

    for idx, (start_date, image_data) in enumerate(filtered_data):
        ax = axes[idx]
        if idx_channel is not None:
            image_data = image_data[idx_channel]
            ax.imshow(image_data, cmap = cmap)  
        else:
            ax.imshow(np.transpose(image_data, (1, 2, 0)))

        ax.axis("off")  
        if idx == 0:
            ax.set_title(f"Original")
        else:
            ax.set_title(f"Aug {idx + 1}")

    # Hide unused subplots
    for ax in axes[num_images:]:
        ax.axis("off")
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_class_distribution(labels, dataset_name="Dataset"):
    """
    Plots the class distribution of the labels.
    :param labels: List or array-like containing class labels.
    :param dataset_name: Name of the dataset (e.g., "Training" or "Validation").
    """
    label_counts = Counter(labels)
    classes = list(label_counts.keys())
    counts = list(label_counts.values())
    
    # Plot
    plt.figure(figsize=(4, 3))
    plt.bar(classes, counts, color='skyblue', edgecolor='black')
    plt.title(f"Class Distribution in {dataset_name}", fontsize=14)
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(classes, fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def resize_dataarray(dataarray):
    
    stacked_data = torch.tensor(dataarray.values, dtype=torch.float32)
  
    stacked_data = stacked_data.unsqueeze(0)  # Shape: [1, 3, 92, 109]

    return F.interpolate(
        stacked_data, 
        size=(224, 224), 
        mode='bicubic', 
        align_corners=False
    ).squeeze(0) 

def normalize_dataarray(dataarray, mean, std):
    stacked_data = torch.tensor(dataarray.values, dtype=torch.float32)

    stacked_data = stacked_data.unsqueeze(0)
    
    normalized_data = stacked_data / 255.0
    
    return (normalized_data - mean[:, None, None]) / std[:, None, None]

def load_netcdf_data(nc_dir, date, dates):
    try:
        if date not in dates.index:
            return None
        
        timestamps = dates.loc[date:].index[:3]
        print(f"Processing Date: {date}, Timestamps: {timestamps}")

        filename_pattern = f"hrv_lr{pd.Timestamp(date).strftime('%Y%m')}.nc"
        matching_files = [f for f in os.listdir(nc_dir) if f == filename_pattern]
        print(f"Matching Files (Pattern 1): {matching_files}")

        if not matching_files:
            filename_pattern = f"hrv_{pd.Timestamp(date).strftime('%Y%m')}.nc"
            matching_files = [f for f in os.listdir(nc_dir) if f == filename_pattern]
            print(f"Matching Files (Pattern 2): {matching_files}")

            if not matching_files:
                raise FileNotFoundError(f"No matching files for date {date}")
           
        filepath = os.path.join(nc_dir, matching_files[0])
        dataset = xr.open_dataset(filepath)

        dataarray = dataset.hrv.sel(time=slice(timestamps[0], timestamps[-1]))
        print(f"DataArray Shape: {dataarray.shape}, IsNull: {dataarray.isnull().any()}")
        if dataarray.shape[0] != 3:
            return None
        if dataarray.isnull().any():
            return None
        
        return dataarray

    except Exception as e:
        print(f"Error Processing Date: {date}, File: {matching_files}, Error: {e}")
        return None
    
def calculate_dataset_mean_std(processed_data):
    all_pixels = []
    for date, resized_image_data in processed_data:
        
        print(f"Processing date {date}, Shape of resized_image_data: {resized_image_data.shape}")
        
        if resized_image_data.shape[0] != 3:
            raise ValueError(f"Expected 3 channels, but got {resized_image_data.shape[0]} for date {date}")

        if isinstance(resized_image_data, xr.DataArray):
            resized_image_data = torch.tensor(resized_image_data.values, dtype=torch.float32)

        all_pixels.append(resized_image_data.view(3, -1))

    all_pixels = torch.cat(all_pixels, dim=1)
    mean = all_pixels.mean(dim=1)
    std = all_pixels.std(dim=1)
    return mean, std

def visualize_random_dates(labels, nc_dir, normalize = True, resize = True, num_samples=3):
    """
    Visualizes random dates labeled as 'cloud_hole' with their image data before and after resizing.
    
    Args:
        labels (str): Path to the labels CSV file.
        nc_dir (str): Directory containing netCDF files.
        num_samples (int): Number of random samples to visualize.
    """

    data = pd.read_csv(labels, index_col=0, parse_dates=True)
    data = data.dropna(subset=["label"])
    data = data[~data["label"].str.contains("problem")]
    data = data[~(pd.DatetimeIndex(data.index).year == 2007)]
    
    years = pd.DatetimeIndex(data.index).year.unique().to_list()
    dates = data[pd.DatetimeIndex(data.index).year.isin(years)]
    dates = dates.sort_index()
    
    cloud_hole_data = data[data["label"] == "cloud_hole"]
    
    processed_data = [
        (date, image_data)
        for date in dates.index
        if (image_data := load_netcdf_data(nc_dir, date, cloud_hole_data)) is not None
    ]

    mean, std = calculate_dataset_mean_std(processed_data)
    mean = mean[:, None, None]
    std = std[:, None, None]

    if len(cloud_hole_data) == 0:
        print("No 'cloud_hole' labeled data found in the dataset.")
        return

    sampled_dates = random.sample(list(cloud_hole_data.index), min(num_samples, len(cloud_hole_data)))
    print(f"Sampled Dates with 'cloud_hole' label: {sampled_dates}")

    for date in sampled_dates:
        dataarray = load_netcdf_data(nc_dir, date, cloud_hole_data)
        
        if dataarray is not None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            original_np = dataarray.values[0]
            axes[0].imshow(original_np, cmap=cmap, vmin=original_np.min(), vmax=original_np.max())
            axes[0].set_title(f"Original Data ({date} Shape {original_np.shape})")
            axes[0].axis("off")
            
            if resize:
                data = resize_dataarray(dataarray)

            if normalize:  
                data = normalize_dataarray(data, mean, std)
            
            data_np = data.permute(1, 2, 0).numpy()
                
            axes[1].imshow(data_np, cmap=cmap, vmin=data_np.min(), vmax=data_np.max())
            axes[1].set_title(f"Resized Data ({date}) Shape {data_np.shape}")
            axes[1].axis("off")
            
            plt.tight_layout()
            plt.show()
        else:
            print(f"No data available for Date: {date}")

