import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import math


def plot_subplots(
    filtered_data, 
    title="Augmented Images",
    figsize=(10, 10),
    idx_channel=None,
    cmap=plt.get_cmap('Greys_r')
):
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

    cols = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()  # Flatten axes for easy iteration

    for idx, (start_date, image_data) in enumerate(filtered_data):
        ax = axes[idx]
        if idx_channel is not None:
            image_data = image_data[idx_channel]
            ax.imshow(image_data, cmap=cmap)
        else:
            ax.imshow(np.transpose(image_data, (1, 2, 0)))

        ax.axis("off")
        if idx == 0:
            ax.set_title("Original")
        else:
            ax.set_title("Aug {idx + 1}")

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
    :param dataset_name: Name of the dataset (e.g., "Training" or "Validation")
    """
    label_counts = Counter(labels)
    classes = list(label_counts.keys())
    counts = list(label_counts.values())

    plt.figure(figsize=(4, 3))
    plt.bar(classes, counts, color='skyblue', edgecolor='black')
    plt.title(f"Class Distribution in {dataset_name}", fontsize=14)
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(classes, fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
