"""Tests for the example_data module."""

from pathlib import Path
from weathercop import example_data


def test_get_example_dataset_path_returns_path_object():
    """get_example_dataset_path should return a Path object."""
    path = example_data.get_example_dataset_path()
    assert isinstance(path, Path)


def test_get_example_dataset_path_file_exists():
    """get_example_dataset_path should point to an existing file."""
    path = example_data.get_example_dataset_path()
    assert path.exists(), f"Example dataset not found at {path}"


def test_get_example_dataset_path_is_netcdf():
    """get_example_dataset_path should point to a .nc file."""
    path = example_data.get_example_dataset_path()
    assert path.suffix == ".nc", f"Expected .nc file, got {path.suffix}"


def test_get_example_dataset_path_is_absolute():
    """get_example_dataset_path should return an absolute path."""
    path = example_data.get_example_dataset_path()
    assert path.is_absolute(), "Path should be absolute for reproducibility"
