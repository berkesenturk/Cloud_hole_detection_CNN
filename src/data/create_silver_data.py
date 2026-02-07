import argparse
from pathlib import Path

from src.utils import NetCDFToZarrConverter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert NetCDF files to Zarr and compute stats"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to raw data folder"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to processed data folder"
    )

    return parser.parse_args()


def main():

    args = parse_args()

    raw_data_path = Path(args.input)
    processed_data_path = Path(args.output)

    seviri_raw = raw_data_path
    seviri_processed = processed_data_path

    input_pattern = "hrv_lr2*.nc"

    print("\n Scanning NetCDF files...")

    files = sorted(seviri_raw.glob(input_pattern))

    if not files:
        raise FileNotFoundError(
            f"No NetCDF files found in {seviri_raw}"
        )

    converter = NetCDFToZarrConverter(
        raw_data_path=seviri_raw,
        processed_data_path=seviri_processed
    )
    chunk_analysis = converter.analyze_netcdf_files(
        input_pattern
    )

    first_year = files[0].as_posix()[-9:-5]
    last_year = files[-1].as_posix()[-9:-5]

    print(f"\n Year range detected: {first_year} → {last_year}")

    zarr_path = (
        seviri_processed /
        f"hrv_lr{first_year}_{last_year}.zarr"
    )
    converter.convert_multiple_files_to_single_zarr(
        file_pattern=input_pattern,
        output_path=zarr_path,
        custom_chunks=chunk_analysis["recommended_chunks"]
    )

# -------------------------------------------------
# Entry point
# -------------------------------------------------


if __name__ == "__main__":
    main()
