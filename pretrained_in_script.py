 
# ## Description
# 
# Here we implement Transfer Learning by using pretrained resnet18 model from Huggingface.


# ## Notes
# 
# #### should test performed with best results of each fold or trained model or both?
# 


import torch
from torch.utils.data import Subset, Dataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoModelForImageClassification, ResNetForImageClassification
from torchvision import transforms
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

cmap = plt.get_cmap('Greys_r')

import os
from concurrent.futures import ThreadPoolExecutor

from utils import (
    plot_subplots,
    plot_class_distribution,
    FocalLoss
)


class CloudHoleDataset(Dataset):
    def __init__(self, labels, nc_dir="./sat_data", train=True, years=None, mean = None, std = None, min = None, max = None, standard_normalize=False):
        """
        Description:
        
        Here preprocessing of the dataset is done.

        Parameters:

        labels: str
            Path to the CSV file containing the labels and dates.
        nc_dir: str
            Path to the directory containing the netCDF files.
        train: bool
            Whether to use the training set or validation set.
        years: list
            List of years to include in the dataset.
        mean: float
            Mean value of the dataset for standard normalization.
        std: float 
            Standard deviation value of the dataset for standard normalization.
        min: float
            Minimum value of the dataset min-max normalization.
        max: float
            Maximum value of the dataset for normalization.
        standard_normalize: bool
            Whether to use standard normalization or min-max normalization.
        
        """
        self.labels = labels
        self.train = train
        self.nc_dir = nc_dir
        self.standard_normalize = standard_normalize
        
        self.data = pd.read_csv(labels, index_col=0, parse_dates=True)

        self.mean = mean
        self.std = std
        self.min = min
        self.max = max
        
        self.data = self.data.dropna(subset=["label"])
        self.data = self.data[~self.data["label"].str.contains("problem")]

        # 2007 is also added by commenting out the line below.
        # self.data = self.data[~(pd.DatetimeIndex(self.data.index).year == 2007)]

        self.dates = self.data[pd.DatetimeIndex(self.data.index).year.isin(years)]
        self.dates = self.dates.sort_index()
        
        # berke TODO: paralellize this process as well
        # loading nc datasets
        self.ds_list = [
            (start_date, image_data)
            for start_date in self.dates.index
            if (image_data := self._load_netcdf_data(start_date)) is not None
        ]
        
        # resizing ds to 224x224. we also achieve a uniform shape for the images.
        self.ds_list_resized = [
            (start_date, image_data)
            for (start_date, dataarray) in self.ds_list
            if (image_data := self._resize_datarray(dataarray)) is not None
        ]
        if (self.mean and self.std) is None:
            self.mean, self.std = self._calculate_dataset_mean_std()
        if (self.min and self.max) is None:
            self.max, self.min = self._calculate_dataset_min_max()
        
        # normalizing ds
        self.ds_list_resized_normalized = [
            (start_date, image_data)
            for (start_date, dataarray) in self.ds_list_resized
            if (image_data := self._normalize_dataarray(dataarray)) is not None
        ]
           
        if self.train:
            augmented_data = []

            # here we apply augmentations to the cloud hole images only
            # we implement parallel processing to speed up the process
            def process_augmentation(args):
                start_date, image_data = args
                label_row = self.data.loc[start_date]
                if label_row["label"] == "cloud_hole":
                    augmented_images = self._apply_augmentations(image_data)
                    return [(start_date, img) for img in augmented_images]
                return []

            with ThreadPoolExecutor() as executor:
                results = executor.map(process_augmentation, self.ds_list_resized_normalized)

            for result in results:
                augmented_data.extend(result)

            self.ds_list_resized_normalized.extend(augmented_data)

    def _resize_datarray(self, dataarray: xr.DataArray) -> torch.tensor:
        """
        Description:
        Resize the given DataArray to 224x224 using bicubic interpolation.
        
        Parameters:
        dataarray: xr.DataArray
            The input DataArray to resize.
        
        Returns: torch.Tensor: The resized tensor.
        """
        try:
            data_tensor = torch.tensor(dataarray.values, dtype=torch.float32)

            if data_tensor.ndim == 3:  # (C, H, W)
                data_tensor = data_tensor.unsqueeze(0)  # (1, C, H, W)

            resized_tensor = F.interpolate(
                data_tensor,
                size=(224, 224),
                mode='bicubic',
                align_corners=False,
            )
            
            resized_tensor = resized_tensor.squeeze(0)  # (C, 224, 224)
            
            return resized_tensor
         
        except Exception as e:
            print(f"Error in resizing DataArray: {e}")
            return None
            
    def _calculate_dataset_min_max(self):
        """
        Description:
        Calculate the min and max of the entire dataset.
        
        """
        return min(da[1].min().item() for da in self.ds_list_resized), max(da[1].max().item() for da in self.ds_list_resized)
    
    def _normalize_dataarray(self, resized_tensor: torch.tensor) -> torch.tensor:
        """
        Description:
        Normalize a dataarray by using either standard or min max normalization.

        Parameters:
            torch.Tensor: The input resized (224x224) data to normalize.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        try:
            
            if isinstance(self.mean, (list, tuple)):
                self.mean = torch.tensor(self.mean, dtype=torch.float32)
            if isinstance(self.std, (list, tuple)):
                self.std = torch.tensor(self.std, dtype=torch.float32)

            if self.standard_normalize:
                normalized_data = (resized_tensor - self.mean) / self.std
            else: 
                normalized_data = (resized_tensor - self.min) / (self.max - self.min)
            return normalized_data

        except Exception as e:
            print(f"Error normalizing dataarray: {e}")
            raise

    def _load_netcdf_data(self, date) -> xr.DataArray:
        """
        Description:
            Load the netCDF file corresponding to the given date and consecutive two timesteps.
            If the file is not found, return None.
        Parameters:
            date: str
                The date for which to load the netCDF file.
        """
        try:
            timestamps = self.dates.loc[date:].index[:3] # Get the first 3 timestamps for the start date

            directory = self.nc_dir

            filename_pattern = f"hrv_lr{pd.Timestamp(date).strftime('%Y%m')}.nc"
            matching_files = [f for f in os.listdir(directory) if f == filename_pattern]

            if not matching_files:
                filename_pattern = f"hrv_{pd.Timestamp(date).strftime('%Y%m')}.nc"
                matching_files = [f for f in os.listdir(directory) if f == filename_pattern]

                if not matching_files:
                    raise FileNotFoundError()

            filepath = os.path.join(directory, matching_files[0])
            dataset = xr.open_dataset(filepath)

            dataarray = dataset.hrv.sel(time=slice(timestamps[0], timestamps[-1]))
            if dataarray.shape[0] != 3:
                return None

            if dataarray.isnull().any():
                return None

            return dataarray

        except Exception as e:
            print(f"File: {matching_files} Date: {date} Unexpected error: {e}")
            return None

    def _apply_augmentations(self, image):
        """
        Apply random augmentations to the given image tensor.
        Parameters:
            image: torch.Tensor
                The image tensor to augment.
        Returns:
            list[torch.Tensor]: A list of augmented image tensors.
        """
        augmented_images = []

        # Convert DataArray to Torch Tensor if not already
        if isinstance(image, xr.DataArray):
            image_tensor = torch.tensor(image.values, dtype=torch.float32)
            if image_tensor.ndim == 3:  # Ensure shape is (C, H, W) for augmentations
                pass
            elif image_tensor.ndim == 2:  # Convert grayscale (H, W) to (1, H, W)
                image_tensor = image_tensor.unsqueeze(0)
            else:
                raise ValueError(f"Unexpected tensor shape: {image_tensor.shape}")
            
        elif isinstance(image, torch.Tensor):
            image_tensor = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # augmentation_pipelines = [
        #     transforms.Compose([
        #         transforms.RandomHorizontalFlip(),
        #         transforms.RandomRotation(10),
        #         # transforms.Normalize(mean=self.mean, std=self.std),
        #     ]),
        #     transforms.Compose([
        #         transforms.RandomVerticalFlip(),
        #         transforms.GaussianBlur(3),
        #         # transforms.Normalize(mean=self.mean, std=self.std),
        #     ]),
        #     transforms.Compose([
        #         transforms.RandomAffine(degrees=10),
        #         transforms.RandomRotation(5),
        #         # transforms.Normalize(mean=self.mean, std=self.std),
        #     ]),
        # ]
        augmentation_pipelines = [
            transforms.Compose([
                transforms.RandomHorizontalFlip(p = 1),
                # transforms.RandomRotation(10),
                # transforms.Normalize(mean=self.mean, std=self.std),
            ]),
            transforms.Compose([
                transforms.RandomVerticalFlip(p = 1),
                # transforms.GaussianBlur(3),
                # transforms.Normalize(mean=self.mean, std=self.std),
            ]),
            transforms.Compose([
                # transforms.RandomAffine(degrees=10),
                transforms.RandomRotation(5),
                # transforms.Normalize(mean=self.mean, std=self.std),
            ]),
        ]
        for pipeline in augmentation_pipelines:
            augmented_image = pipeline(image_tensor)
            if not torch.isnan(augmented_image).any():
                augmented_images.append(augmented_image)

        return augmented_images

    def _calculate_dataset_mean_std(self):
        """
        Calculate the mean and standard deviation of the entire dataset.
        """
        all_pixels = []

        for date, resized_image_data in self.ds_list_resized:
            if isinstance(resized_image_data, xr.DataArray):
                resized_image_data = torch.tensor(resized_image_data.values, dtype=torch.float32)

            all_pixels.append(resized_image_data.view(3, -1))

        all_pixels = torch.cat(all_pixels, dim=1)  

        mean = all_pixels.mean()
        std = all_pixels.std()

        # print(f"Calculated mean: {mean}")
        # print(f"Calculated std: {std}")

        return mean, std
    
    def __len__(self):
        return len(self.ds_list_resized_normalized)
    
    def __getitem__(self, idx):
        """       
        Description:
        
            Retrieves the data sample corresponding to the given index.
            - Combines image tensors into a batch.
            - Determines and returns the label tensor.
        Parameters:
            idx: int
                The index of the dataset to retrieve.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the image tensor and label tensor.
        """
        start_date, image_data_list = self.ds_list_resized_normalized[idx]
        image_tensors = []

        for image_data in image_data_list:
            
            if torch.isnan(image_data).any():
                print(f"NaN values might be found for the dates: {self.dates.loc[start_date:].index[:3]} in the raw image data")
            image_tensors.append(image_data)

        images_tensor = torch.stack(image_tensors)
        label_row = self.data.loc[start_date]
        label = 1 if label_row['label'] == 'cloud_hole' else 0
        label_tensor = torch.tensor(label, dtype=torch.long)

        return images_tensor, label_tensor
    


# berke TODO: which dates are not augmentated? why? 51 missing meaning 17 datetime did not augmentated

TRAIN_YEARS = [2005, 2007, 2010, 2013, 2016]
TEST_YEARS = [2015, 2017] # TODO: find other test years
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = CloudHoleDataset(
    labels = "./labels.csv",
    nc_dir = "./sat_data", 
    train = True, 
    years = TRAIN_YEARS
    # years = pd.DatetimeIndex(labels.index).year.unique().to_list()
)

print(dataset.mean, dataset.std, dataset.min, dataset.max)

test_dataset = CloudHoleDataset('./labels.csv', './sat_data', False, years = TEST_YEARS, mean = dataset.mean, std = dataset.std, min = dataset.min, max = dataset.max)

 
# ### Visualization of Original and Augmented Images
# 
# Visualizing for controlling how augmentation works or to inspect suspicious data from prediction results.


desired_date = "2007-02-19 11:00:00"

# Filter augmented and original images for the desired date
filtered_resized_data = [
    (start_date, image_data)
    for start_date, image_data in dataset.ds_list_resized_normalized
    if str(start_date) == desired_date
]

print(f"Number of images (original + augmentations) for {desired_date}: {len(filtered_resized_data)}")

# Example: Access augmented image shapes
for idx, (start_date, image_data) in enumerate(filtered_resized_data):
    print(f"Augmentation {idx + 1}: Image Shape: {image_data.shape}")
    
plot_subplots(filtered_resized_data, title=f"Augmented Images for start_date: {desired_date}", figsize=(10, 10), idx_channel=1)

 
# ### plot class distribution


from collections import Counter

_train_dataloader_one_batch = DataLoader(dataset, batch_size=1, shuffle=False)

all_train_labels = []

for _, label_tensor in _train_dataloader_one_batch:
    all_train_labels.append(label_tensor.item())

plot_class_distribution(all_train_labels, dataset_name="Training Set")

print("dataset Distribution:", Counter(all_train_labels))


 
# ### configuration of hyperparameters
# 


BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 1e-3
NUM_SPLITS_TSCV = 5
EARLY_STOPPING_PATIENCE = 5


 
# ### Training via Transfer Learning by using pretrained resnet 18


# TODO berke: move all functions to classes for transferlearning, plots and so forth.

model_pretrained = ResNetForImageClassification.from_pretrained(
    "microsoft/resnet-18",
    num_labels = 2,
    ignore_mismatched_sizes=True
)

# Freeze all layers except the classifier
for param in model_pretrained.parameters():
    param.requires_grad = False

# Unfreeze the classifier layers
for param in model_pretrained.classifier.parameters():
    param.requires_grad = True

model_pretrained.to(DEVICE)

def train_model(model, train_loader, criterion, optimizer):
    """
    Description:
        Train the model using the given training data.
    Parameters:
        model: ResNetForImageClassification
            The model to train.
        train_loader: DataLoader
            The training data loader.
        criterion: nn.Module
            The loss function.
        optimizer: optim.Optimizer
            The optimizer to use for training.
    Returns:
        float: The average training loss.
        float: The training accuracy
    """
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        _, predicted = torch.max(outputs.logits, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_accuracy = 100 * correct / total
    
    return train_loss, train_accuracy

def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total
    
    return val_loss, val_accuracy

def early_stopping(val_loss, best_val_loss, counter, patience):
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping triggered")
    return best_val_loss, counter

def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, fold):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, 'r-', label='Training Accuracy')
    plt.plot(val_accuracies, 'b-', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Fold {fold + 1} - Training vs Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, 'r-', label='Training Loss')
    plt.plot(val_losses, 'b-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Fold {fold + 1} - Training vs Validation Loss')
    plt.legend()
    plt.grid(True) 
    
    plt.tight_layout()
    plt.savefig(f'training_curves_fold_{fold + 1}.png')
    plt.close()

from torch.optim.lr_scheduler import ReduceLROnPlateau
criterion = FocalLoss(alpha=0.01, gamma=10, num_classes=2)
optimizer = optim.Adam(model_pretrained.parameters(), lr=LEARNING_RATE)

# Initialize ReduceLROnPlateau scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)


def validation_of_pretrained_model(
    batch_size,
    scheduler,
    num_epochs,
    num_splits,
    patience,
    model 
):
    """
    Description:
        Perform time-series cross-validation on the pretrained model using the given parameters.

    Parameters:    
        batch_size: int
            The batch size for training.
        learning_rate: float
            The learning rate for training.
        num_epochs: int
            The number of epochs to train for.
        num_splits: int
            The number of splits for time-series cross-validation.
        patience: int
            The number of epochs to wait before stopping if no improvement.
    
    """
    tscv = TimeSeriesSplit(n_splits=num_splits)
    fold_metrics = []
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    count_epochs = 0
    for fold, (train_idx, val_idx) in enumerate(tscv.split(range(len(dataset)))):
        print(f"Fold {fold + 1}/{num_splits}")

        # print("\tTRAIN indices:", train_idx)
        # print("\tVALIDATION indices:", val_idx)

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        best_val_loss = float('inf')
        counter = 0

        for epoch in range(num_epochs):
            train_loss, train_accuracy = train_model(model, train_loader, criterion, optimizer)
            val_loss, val_accuracy = validate_model(model, val_loader, criterion)
            
            scheduler.step(val_loss)

            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}%")

            best_val_loss, counter = early_stopping(val_loss, best_val_loss, counter, patience)

            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                
                plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, fold)
                break

            # Save the best model 
            if val_loss == best_val_loss:
                torch.save({
                    'fold': fold + 1,
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy
                }, f'best_model_fold_{fold + 1}.pth')

        plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, fold)

        fold_metrics.append((val_loss, val_accuracy))

    avg_val_loss = np.mean([metrics[0] for metrics in fold_metrics])
    avg_val_accuracy = np.mean([metrics[1] for metrics in fold_metrics])
    folds = range(1, num_splits + 1)
    val_loss_per_fold = [metrics[0] for metrics in fold_metrics]
    val_acc_per_fold = [metrics[1] for metrics in fold_metrics]

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.bar(folds, val_loss_per_fold, color='skyblue')
    plt.xlabel("Fold")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss Across Folds")

    plt.subplot(1, 2, 2)
    plt.bar(folds, val_acc_per_fold, color='lightgreen')
    plt.xlabel("Fold")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Validation Accuracy Across Folds")

    plt.tight_layout()
    plt.show()

    print(f"Average Validation Loss: {avg_val_loss:.4f}")
    print(f"Average Validation Accuracy: {avg_val_accuracy:.2f}%")

    return train_losses, val_losses, train_accuracies, val_accuracies, fold_metrics, model

train_losses, val_losses, train_accuracies, val_accuracies, fold_metrics, model_pretrained = validation_of_pretrained_model(
    batch_size=BATCH_SIZE,
    scheduler=scheduler,
    num_epochs=EPOCHS,
    num_splits=NUM_SPLITS_TSCV,
    patience=EARLY_STOPPING_PATIENCE,
    model = model_pretrained
)


folds = range(1, NUM_SPLITS_TSCV + 1)
val_loss_per_fold = [metrics[0] for metrics in fold_metrics]
val_acc_per_fold = [metrics[1] for metrics in fold_metrics]

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.bar(folds, val_loss_per_fold, color='skyblue')
plt.xlabel("Fold")
plt.ylabel("Validation Loss")
plt.title("Validation Loss Across Folds")

plt.subplot(1, 2, 2)
plt.bar(folds, val_acc_per_fold, color='lightgreen')
plt.xlabel("Fold")
plt.ylabel("Validation Accuracy (%)")
plt.title("Validation Accuracy Across Folds")

plt.tight_layout()
plt.show()


 
# ## Testing with Best results of each Folds: they're saved as best_model_fold_*.pth


def evaluate_model(model, test_loader, criterion, device):
    """Evaluate the model on the test set."""
    test_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            test_loss += criterion(outputs, labels).item()

            _, predicted = torch.max(outputs.logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    test_loss /= len(test_loader)
    test_accuracy = 100. * correct / total

    return test_loss, test_accuracy, all_labels, all_predictions

def evaluate_folds(model, test_dataset, criterion, device, num_splits_tscv, batch_size):
    """Evaluate the model across all folds and return metrics."""
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    fold_test_metrics = []  # To store metrics for each fold
    all_test_labels = []
    all_test_predictions = []

    # Loop through all folds
    for fold in range(num_splits_tscv):
        print(f"Evaluating Test Set with Best Model of Fold {fold + 1}")
        
        checkpoint = torch.load(f"best_model_fold_{fold + 1}.pth")
        model.load_state_dict(checkpoint['model_state_dict'])   
        model.eval()
        model.to(device)

        # Evaluate the model
        test_loss, test_accuracy, fold_labels, fold_predictions = evaluate_model(model, test_loader, criterion, device)
        
        print(f"Fold {fold + 1} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        
        # Save metrics for this fold
        fold_test_metrics.append((test_loss, test_accuracy))
        all_test_labels.extend(fold_labels)
        all_test_predictions.extend(fold_predictions)

    return fold_test_metrics, all_test_labels, all_test_predictions

def compute_overall_metrics(fold_test_metrics):
    """Compute overall metrics across all folds."""
    avg_test_loss = np.mean([metrics[0] for metrics in fold_test_metrics])
    avg_test_accuracy = np.mean([metrics[1] for metrics in fold_test_metrics])
    return avg_test_loss, avg_test_accuracy

def plot_confusion_matrix(all_test_labels, all_test_predictions, class_names):
    """Plot confusion matrix."""
    conf_matrix = confusion_matrix(all_test_labels, all_test_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Overall)")
    plt.show()

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

CLASS_NAMES = ["Non Cloud Hole", "Cloud Hole"]  # Class names for classification report and confusion matrix
criterion = FocalLoss(alpha=0.01, gamma=10, num_classes=2)

fold_test_metrics = []  # To store metrics for each fold
all_test_labels = []
all_test_predictions = []

fold_test_metrics, all_test_labels, all_test_predictions = evaluate_folds(
    model_pretrained, test_dataset, criterion, DEVICE, NUM_SPLITS_TSCV, BATCH_SIZE
)

avg_test_loss, avg_test_accuracy = compute_overall_metrics(fold_test_metrics)

print(f"\nOverall Test Loss (Average across folds): {avg_test_loss:.4f}")
print(f"Overall Test Accuracy (Average across folds): {avg_test_accuracy:.2f}%")

print("\nOverall Classification Report:")
print(classification_report(all_test_labels, all_test_predictions, target_names=CLASS_NAMES))

plot_confusion_matrix(all_test_labels, all_test_predictions, CLASS_NAMES)
print(model_pretrained.state_dict())

torch.save(model_pretrained.state_dict(), "final_model.pth")
print("Final model trained and saved as final_model.pth")
