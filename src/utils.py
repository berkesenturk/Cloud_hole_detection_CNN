import logging
from pathlib import Path
from typing import Dict, Optional

import xarray as xr
import zarr
from dask.diagnostics import ProgressBar


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class NetCDFToZarrConverter:
    """
    Definition:
        Converts NetCDF files to Zarr format optimized
        for cloud storage and ML training.
    """

    def __init__(
        self,
        chunk_time: int = 10,
        chunk_spatial: int = 256,
        compression_level: int = 3,
        use_dask: bool = True,
        raw_data_path: Path = None,
        processed_data_path: Path = None
    ):
        """
        Initialize converter with optimal chunking parameters

        Args:
            chunk_time: Number of time steps per chunk
            chunk_spatial: Spatial dimension chunk size (lat/lon)
            compression_level: Compression level (1-9, 3 recommended)
            use_dask: Use dask for parallel processing
            raw_data_path: Path to raw data folder
            processed_data_path: Path to processed data folder
        """
        self.chunk_time = chunk_time
        self.chunk_spatial = chunk_spatial
        self.compression_level = compression_level
        self.use_dask = use_dask
        self.raw_data_path = Path(raw_data_path) if raw_data_path else None
        self.processed_data_path = (
            Path(processed_data_path) if processed_data_path else None
        )

    def analyze_netcdf_files(self, file_pattern: str) -> Dict:
        """
        Analyze NetCDF files to determine optimal chunking strategy

        Args:
            file_pattern: Glob pattern for NetCDF files
                (e.g., "data/hrv_lr*.nc")

        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Analyzing NetCDF files matching: {file_pattern}")

        files = sorted(self.raw_data_path.glob(file_pattern))

        if not files:
            raise ValueError(
                f"No files found matching pattern: {file_pattern}"
            )

        logger.info(f"Found {len(files)} files")

        # Open files using resolved paths
        with xr.open_mfdataset(files, combine="by_coords") as ds:
            dims = dict(ds.dims)
            variables = list(ds.data_vars)
            coords = list(ds.coords)

            # Calculate approximate chunk size in MB
            sample_var = ds[variables[0]]
            dtype_size = sample_var.dtype.itemsize

        analysis = {
            "num_files": len(files),
            "files": [f.name for f in files],
            "dimensions": dims,
            "variables": variables,
            "coordinates": coords,
            "dtype_size": dtype_size,
            "recommended_chunks": self._calculate_optimal_chunks(
                dims, dtype_size
            ),
        }

        logger.info(f"Analysis complete: {analysis}")
        return analysis

    def _calculate_optimal_chunks(
        self, dims: Dict, dtype_size: int
    ) -> Dict:
        """
        Calculate optimal chunk sizes for Zarr storage.
        Target: ~20 MB per chunk.
        Only scales the time dimension.
        Spatial dimensions remain full size.
        """

        target_chunk_bytes = 20 * 1024 * 1024  # 20 MB

        # Identify dimension names
        time_dim = next((d for d in ["time"] if d in dims), None)
        lat_dim = next(
            (d for d in ["lat", "latitude", "y"] if d in dims), None
        )
        lon_dim = next(
            (d for d in ["lon", "longitude", "x"] if d in dims), None
        )

        if not (time_dim and lat_dim and lon_dim):
            raise ValueError(
                "Expected time, lat/y, lon/x dimensions for Zarr chunking"
            )

        t_size = dims[time_dim]
        lat_size = dims[lat_dim]
        lon_size = dims[lon_dim]

        # Full spatial chunk
        lat_chunk = lat_size
        lon_chunk = lon_size

        # Bytes per single timestep
        bytes_per_timestep = lat_chunk * lon_chunk * dtype_size

        if bytes_per_timestep == 0:
            raise ValueError(
                "Invalid dimension sizes for chunk calculation"
            )

        # Compute time chunk
        t_chunk = int(target_chunk_bytes / bytes_per_timestep)

        # Safety clamps
        t_chunk = max(1, min(t_chunk, t_size))

        chunks = {
            time_dim: t_chunk,
            lat_dim: lat_chunk,
            lon_dim: lon_chunk,
        }

        # Logging
        chunk_elements = t_chunk * lat_chunk * lon_chunk
        chunk_mb = (chunk_elements * dtype_size) / (1024 * 1024)

        logger.info(
            f"Calculated chunk size: {chunk_mb:.2f} MB | chunks={chunks}"
        )

        return chunks

    def convert_single_file(
        self,
        input_file: str,
        output_path: str,
        custom_chunks: Optional[Dict] = None,
    ):
        """
        Convert a single NetCDF file to Zarr
        """
        input_path = (
            self.raw_data_path / input_file
            if self.raw_data_path and not Path(input_file).is_absolute()
            else Path(input_file)
        )

        output_path = (
            self.processed_data_path / output_path
            if self.processed_data_path
            and not Path(output_path).is_absolute()
            else Path(output_path)
        )

        logger.info(f"Converting {input_path} to {output_path}")

        try:
            if self.use_dask:
                ds = xr.open_dataset(input_path, chunks="auto")
            else:
                ds = xr.open_dataset(input_path)

            if custom_chunks:
                chunks = custom_chunks
            else:
                chunks = self._calculate_optimal_chunks(
                    dict(ds.dims),
                    ds[list(ds.data_vars)[0]].dtype.itemsize,
                )

            if self.use_dask:
                ds = ds.chunk(chunks)

            encoding = self._get_encoding(ds, chunks)

            logger.info(f"Writing to Zarr with chunks: {chunks}")

            if self.use_dask:
                with ProgressBar():
                    ds.to_zarr(
                        output_path,
                        mode="w",
                        consolidated=True,
                        encoding=encoding,
                    )
            else:
                ds.to_zarr(
                    output_path,
                    mode="w",
                    consolidated=True,
                    encoding=encoding,
                )

            ds.close()
            logger.info(f"Successfully converted {input_path}")

        except Exception as e:
            logger.error(f"Error converting {input_path}: {str(e)}")
            raise

    def convert_multiple_files_to_single_zarr(
        self,
        file_pattern: str,
        output_path: str,
        custom_chunks: Optional[Dict] = None,
    ):
        """
        Convert multiple NetCDF files to a single consolidated Zarr store
        """

        logger.info(
            f"Converting multiple files to single Zarr: {file_pattern}"
        )

        files = sorted(self.raw_data_path.glob(file_pattern))

        if not files:
            raise ValueError(f"No files found matching: {file_pattern}")

        logger.info(f"Found {len(files)} files to convert")

        try:
            ds = xr.open_mfdataset(files, combine="by_coords")

            if custom_chunks:
                chunks = custom_chunks
            else:
                chunks = self._calculate_optimal_chunks(
                    dict(ds.dims),
                    ds[list(ds.data_vars)[0]].dtype.itemsize,
                )

            if self.use_dask:
                ds = ds.chunk(chunks)

            encoding = self._get_encoding(ds, chunks)


            logger.info(
                f"Writing consolidated Zarr with chunks: {chunks}"
            )
            print(ds.nbytes / (1024 ** 3), "GB dataset")
            print(ds.chunks)

            if self.use_dask:
                with ProgressBar():
                    ds.to_zarr(
                        output_path,
                        mode="w",
                        consolidated=True,
                        encoding=encoding,
                    )
            else:
                ds.to_zarr(
                    output_path,
                    mode="w",
                    consolidated=True,
                    encoding=encoding,
                )

            ds.close()
            logger.info(
                f"Successfully created consolidated Zarr at {output_path}"
            )

        except Exception as e:
            logger.error(f"Error during conversion: {str(e)}")
            raise

    def convert_files_individually(
        self,
        file_pattern: str,
        output_dir: str,
        custom_chunks: Optional[Dict] = None,
    ):
        """Convert each NetCDF file to a separate Zarr store"""

        files = sorted(self.raw_data_path.glob(file_pattern))
        output_path = Path(self.processed_data_path) / output_dir
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Converting {len(files)} files individually")

        for i, file in enumerate(files, 1):
            zarr_name = file.stem + ".zarr"
            zarr_path = output_path / zarr_name

            logger.info(f"[{i}/{len(files)}] Converting {file.name}")

            self.convert_single_file(
                str(file),
                str(zarr_path),
                custom_chunks,
            )

    def _get_encoding(self, ds: xr.Dataset, chunks: Dict) -> Dict:
        """Generate optimal encoding with compression for all variables"""

        encoding = {}

        for var in ds.data_vars:
            encoding[var] = {
                "compressor": zarr.Blosc(
                    cname="zstd",
                    clevel=self.compression_level,
                    shuffle=zarr.Blosc.BITSHUFFLE,
                ),
                "chunks": tuple(
                    chunks.get(dim, ds[var].shape[i])
                    for i, dim in enumerate(ds[var].dims)
                ),
            }

        return encoding

    def _print_zarr_info(self, zarr_path: str):
        """Print information about the created Zarr store"""

        zarr_path = (
            self.processed_data_path / zarr_path
            if self.processed_data_path
            and not Path(zarr_path).is_absolute()
            else Path(zarr_path)
        )

        z = zarr.open(zarr_path, mode="r")
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Zarr Store Information: {zarr_path}")
        logger.info(f"{'=' * 60}")
        logger.info(z.tree())
        logger.info(f"{'=' * 60}\n")
