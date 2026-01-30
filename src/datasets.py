import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms
import pandas as pd
import xarray as xr

import os
from concurrent.futures import ThreadPoolExecutor


class CloudHoleDataset(Dataset):
    def __init__(
        self, labels, nc_dir="./sat_data", train=True, years=None,
        mean=None, std=None, min=None, max=None,
        standard_normalize=False
    ):
        """
        Description:

        Initializes the CloudHoleDataset with desired preprocessing steps.

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
        # self.data = self.data[
        # ~(pd.DatetimeIndex(self.data.index).year == 2007)
        # ]

        self.dates = self.data[
            pd.DatetimeIndex(self.data.index).year.isin(years)
        ]
        self.dates = self.dates.sort_index()

        # berke TODO: paralellize this process as well
        # loading nc datasets
        self.ds_list = [
            (start_date, image_data)
            for start_date in self.dates.index
            if (image_data := self._load_netcdf_data(start_date)) is not None
        ]

        # resizing ds to 224x224. we also achieve a uniform shape
        # for the images.
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

            # here we apply augmentations to the cloud hole
            # images only
            # we implement parallel processing to speed up
            # the process
            def process_augmentation(args):
                start_date, image_data = args
                label_row = self.data.loc[start_date]
                if label_row["label"] == "cloud_hole":
                    augmented_images = self._apply_augmentations(
                        image_data
                    )
                    return [(start_date, img) for img in augmented_images]
                return []

            with ThreadPoolExecutor() as executor:
                results = executor.map(
                    process_augmentation,
                    self.ds_list_resized_normalized
                )

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
        min_val = min(da[1].min().item() for da in self.ds_list_resized)
        max_val = max(da[1].max().item() for da in self.ds_list_resized)
        return min_val, max_val

    def _normalize_dataarray(
        self, resized_tensor: torch.tensor
    ) -> torch.tensor:
        """
        Description:
        Normalize a dataarray by using either
        standard or min max normalization.

        Parameters:
            torch.Tensor: The input resized (224x224) data to normalize.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        try:

            if isinstance(self.mean, (list, tuple)):
                self.mean = torch.tensor(
                    self.mean, dtype=torch.float32
                )
            if isinstance(self.std, (list, tuple)):
                self.std = torch.tensor(
                    self.std, dtype=torch.float32
                )

            if self.standard_normalize:
                normalized_data = (
                    (resized_tensor - self.mean) / self.std
                )
            else:
                normalized_data = (
                    (resized_tensor - self.min) / (self.max - self.min)
                )
            return normalized_data

        except Exception as e:
            print(f"Error normalizing dataarray: {e}")
            raise

    def _load_netcdf_data(self, date) -> xr.DataArray:
        """
        Description:
            Load the netCDF file corresponding to the given date and
            consecutive two timesteps.
            If the file is not found, return None.
        Parameters:
            date: str
                The date for which to load the netCDF file.
        """
        try:
            # Get the first 3 timestamps for the start date
            timestamps = self.dates.loc[date:].index[:3]

            directory = self.nc_dir

            filename_pattern = (
                f"hrv_lr{pd.Timestamp(date).strftime('%Y%m')}.nc"
            )
            matching_files = [
                f for f in os.listdir(directory)
                if f == filename_pattern
            ]

            if not matching_files:
                filename_pattern = (
                    f"hrv_{pd.Timestamp(date).strftime('%Y%m')}.nc"
                )
                matching_files = [
                    f for f in os.listdir(directory)
                    if f == filename_pattern
                ]

                if not matching_files:
                    raise FileNotFoundError()

            filepath = os.path.join(directory, matching_files[0])
            dataset = xr.open_dataset(filepath)

            dataarray = dataset.hrv.sel(
                time=slice(timestamps[0], timestamps[-1])
            )
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
            list[torch.Tensor]: A list of augmented image
                tensors.
        """
        augmented_images = []

        # Convert DataArray to Torch Tensor if not already
        if isinstance(image, xr.DataArray):
            image_tensor = torch.tensor(
                image.values, dtype=torch.float32
            )
            # Ensure shape is (C, H, W) for augmentations
            if image_tensor.ndim == 3:
                pass
            # Convert grayscale (H, W) to (1, H, W)
            elif image_tensor.ndim == 2:
                image_tensor = image_tensor.unsqueeze(0)
            else:
                raise ValueError(
                    f"Unexpected tensor shape: {image_tensor.shape}"
                )

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
                transforms.RandomHorizontalFlip(p=1),
                # transforms.RandomRotation(10),
                # transforms.Normalize(mean=self.mean, std=self.std),
            ]),
            transforms.Compose([
                transforms.RandomVerticalFlip(p=1),
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
                resized_image_data = torch.tensor(
                    resized_image_data.values, dtype=torch.float32
                )

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

            Retrieves the data sample corresponding to the given
            index.
            - Combines image tensors into a batch.
            - Determines and returns the label tensor.
        Parameters:
            idx: int
                The index of the dataset to retrieve.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing
                the image tensor and label tensor.
        """
        start_date, image_data_list = (
            self.ds_list_resized_normalized[idx]
        )
        image_tensors = []

        for image_data in image_data_list:

            if torch.isnan(image_data).any():
                print(
                    f"NaN values might be found for the dates: "
                    f"{self.dates.loc[start_date:].index[:3]} "
                    f"in the raw image data"
                )
            image_tensors.append(image_data)

        images_tensor = torch.stack(image_tensors)
        label_row = self.data.loc[start_date]
        label = 1 if label_row['label'] == 'cloud_hole' else 0
        label_tensor = torch.tensor(label, dtype=torch.long)

        return images_tensor, label_tensor
