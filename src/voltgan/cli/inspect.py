from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import h5py
import numpy as np

from voltgan.config import HDF_DIR


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: voltgan inspect <hdf-rel-path>")

    hdf_path = HDF_DIR / Path(sys.argv[2])
    if not hdf_path.exists():
        raise FileNotFoundError(f"HDF file not found: {hdf_path}")

    print(f"Inspecting {hdf_path}")
    print(f"File size: {hdf_path.stat().st_size / 1e6:.2f} MB")
    print()

    with h5py.File(hdf_path, "r") as f:
        print("Root attributes:")
        _print_attrs(dict(f.attrs))
        print()

        print("Contents:")
        for name, obj in f.items():
            if isinstance(obj, h5py.Group):
                _print_group(name, cast(h5py.Group, obj))
                for sub_name, sub_obj in obj.items():
                    if isinstance(sub_obj, h5py.Dataset):
                        _print_dataset(
                            sub_name, cast(h5py.Dataset, sub_obj), indent="    "
                        )
                    elif isinstance(sub_obj, h5py.Group):
                        _print_group(sub_name, cast(h5py.Group, sub_obj), indent="    ")
            elif isinstance(obj, h5py.Dataset):
                _print_dataset(name, cast(h5py.Dataset, obj))


def _print_attrs(attrs: dict, indent: str = "  ") -> None:
    for key, value in attrs.items():
        if isinstance(value, np.ndarray):
            value = value.tolist()
        print(f"{indent}@ {key} = {value}")


def _print_dataset(name: str, dataset: h5py.Dataset, indent: str = "  ") -> None:
    print(f"{indent}{name}  (dataset)")
    print(f"{indent}  shape: {dataset.shape}  dtype: {dataset.dtype}")
    if np.issubdtype(dataset.dtype, np.number) and dataset.size and dataset.size > 0:
        arr = dataset[:]
        print(
            f"{indent}  min: {np.min(arr):.4f}  max: {np.max(arr):.4f}  "
            f"mean: {np.mean(arr):.4f}"
        )
    _print_attrs(dict(dataset.attrs), indent + "  ")


def _print_group(name: str, group: h5py.Group, indent: str = "") -> None:
    print(f"{indent}{name}  (group, {len(group)} entries)")
    _print_attrs(dict(group.attrs), indent + "  ")

