import pytest
from src.utils import (
    NetCDFToZarrConverter
)

@pytest.fixture
def converter():
    return NetCDFToZarrConverter()


def test_chunks_have_expected_dims_latlon(converter):
    """Test that calculated chunks include 'time', 'lat', and 'lon' dimensions."""
    dims = {"time": 100, "lat": 50, "lon": 60}

    chunks = converter._calculate_optimal_chunks(dims, dtype_size=4)

    assert "time" in chunks
    assert "lat" in chunks
    assert "lon" in chunks


def test_chunks_have_expected_dims_xy(converter):
    """Test that calculated chunks include 'time', 'x', and 'y' dimensions."""
    dims = {"time": 20, "x": 100, "y": 200}

    chunks = converter._calculate_optimal_chunks(dims, dtype_size=4)

    assert "time" in chunks
    assert "x" in chunks
    assert "y" in chunks



def test_time_chunk_never_zero(converter):
    """Test that time chunk is at least 1."""
    dims = {"time": 1, "lat": 5000, "lon": 5000}

    chunks = converter._calculate_optimal_chunks(dims, dtype_size=4)

    assert chunks["time"] >= 1


def test_time_chunk_not_exceed_total_time(converter):
    """Test that time chunk does not exceed total time dimension."""
    dims = {"time": 5, "lat": 10, "lon": 10}

    chunks = converter._calculate_optimal_chunks(dims, dtype_size=4)

    assert chunks["time"] <= dims["time"]


def test_missing_time_dimension_raises(converter):
    """Test that missing time dimension raises ValueError."""
    dims = {"lat": 10, "lon": 10}

    with pytest.raises(ValueError):
        converter._calculate_optimal_chunks(dims, dtype_size=4)


def test_missing_spatial_dimension_raises(converter):
    """Test that missing spatial dimension raises ValueError."""
    dims = {"time": 10, "lat": 10}

    with pytest.raises(ValueError):
        converter._calculate_optimal_chunks(dims, dtype_size=4)


def test_chunk_size_is_reasonable(converter):
    """Test that calculated chunk size is within reasonable bounds (5MB to 50MB)."""
    dims = {"time": 1000000, "lat": 200, "lon": 200}
    dtype_size = 4  # float32

    chunks = converter._calculate_optimal_chunks(dims, dtype_size)

    chunk_bytes = (
        chunks["time"]
        * chunks["lat"]
        * chunks["lon"]
        * dtype_size
    )

    # 5MB – 50MB tolerance
    assert 5 * 1024 * 1024 < chunk_bytes < 50 * 1024 * 1024


def test_chunk_size_small_dataset(converter):
    """Test that for small datasets, chunk sizes equal dimension sizes."""
    dims = {"time": 3, "lat": 5, "lon": 5}
    dtype_size = 4

    chunks = converter._calculate_optimal_chunks(dims, dtype_size)

    assert chunks["time"] == 3
    assert chunks["lat"] == 5
    assert chunks["lon"] == 5


def test_float64_results_smaller_time_chunk(converter):
    """Test that float64 data results in smaller time chunk than float32 for same dimensions."""
    dims = {"time": 1000, "lat": 200, "lon": 200}

    chunks_f32 = converter._calculate_optimal_chunks(dims, dtype_size=4)
    chunks_f64 = converter._calculate_optimal_chunks(dims, dtype_size=8)

    assert chunks_f64["time"] <= chunks_f32["time"]
