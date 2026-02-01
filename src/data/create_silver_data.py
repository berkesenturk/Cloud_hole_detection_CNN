import os
from datetime import datetime
from pathlib import Path

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.utils import (
    NetCDFToZarrConverter
)
from src.data.datasets import CloudHoleDataset

raw_data_path = "../../data/raw"

# Scaling only performed for time dimension
# 20MB per chunk is aimed

converter = NetCDFToZarrConverter()

chunk_analysis = converter.analyze_netcdf_files(f"{raw_data_path}/seviri/hrv_lr2*.nc")

input_file_pattern = f"{raw_data_path}/seviri/hrv_lr2*.nc"

files = sorted(Path().glob(input_file_pattern))

"""
- Issue:Segmentation fault (core dumped)
- Reason: parallel=True, chunks='auto' on xr.open_mfdataset
- Solution: set parallel false and chunks to None
"""

first_year, last_year = files[0].as_posix()[-9:-5], files[-1].as_posix()[-9:-5]

processed_data_path = "../../data/processed"

dataset = CloudHoleDataset(
    labels=f"{processed_data_path}/julia_labels.csv",
    data_dir=f"{processed_data_path}/seviri/hrv_lr{first_year}_{last_year}.zarr",
    years=range(int(first_year), int(last_year)+1)
)

mean, std = dataset.mean, dataset.std

print(f"Mean: {mean}, Std: {std}")