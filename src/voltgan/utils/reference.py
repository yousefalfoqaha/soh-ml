from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from voltgan.config import (
    HDF_ROOT,
    REFERENCE_CURRENT_RANGE,
    REFERENCE_TEMPERATURE_RANGE,
)
from voltgan.utils.discover import discover

_NAN = float("nan")


def select_reference_points(
    records: list[tuple[float, float, float, float]],
    ref_temp_range: tuple[float, float],
    ref_current_range: tuple[float, float],
) -> list[tuple[float, float]]:
    """Return every qualifying (discharge_cycle_index, soh) measurement."""
    ref_lo, ref_hi = ref_temp_range
    cur_lo, cur_hi = ref_current_range

    ref_points: list[tuple[float, float]] = []
    for dci, soh, amb, mni in records:
        if np.isnan(float(soh)):
            continue
        if ref_lo <= amb <= ref_hi and cur_lo <= mni <= cur_hi:
            ref_points.append((dci, soh))
    return ref_points


def load_reference_points(
    mcus: list[str],
    hdf_root: Path = HDF_ROOT,
    ref_temp_range: tuple[float, float] = REFERENCE_TEMPERATURE_RANGE,
    ref_current_range: tuple[float, float] = REFERENCE_CURRENT_RANGE,
) -> list[tuple[float, float]]:
    """Load (discharge_cycle_index, soh) reference points from HDF files."""
    ref_lo, ref_hi = ref_temp_range
    cur_lo, cur_hi = ref_current_range

    points: list[tuple[float, float]] = []
    for hdf_path in discover(hdf_root, mcus, (".hdf",)):
        with h5py.File(hdf_path, "r") as f:
            soh_raw = f.attrs.get("soh", _NAN)
            if np.isnan(float(soh_raw)):
                continue
            dci = float(f.attrs.get("discharge_cycle_index", 0))
            amb = float(f.attrs.get("ambient_temperature", 0))
            if not (ref_lo <= amb <= ref_hi):
                continue
            mni = float(f.attrs.get("mean_neg_current", 0))
            if not (cur_lo <= mni <= cur_hi):
                continue
            points.append((dci, float(soh_raw)))
    points.sort()
    return points


def load_records(
    mcus: list[str],
    hdf_root: Path = HDF_ROOT,
) -> list[tuple[float, float, float, float]]:
    """Load (dci, soh, ambient_temperature, mean_neg_current) for all instances."""
    records: list[tuple[float, float, float, float]] = []
    for hdf_path in discover(hdf_root, mcus, (".hdf",)):
        with h5py.File(hdf_path, "r") as f:
            dci = float(f.attrs.get("discharge_cycle_index", 0))
            soh = float(f.attrs.get("soh", _NAN))
            amb = float(f.attrs.get("ambient_temperature", 0))
            mni = float(f.attrs.get("mean_neg_current", 0))
        records.append((dci, soh, amb, mni))
    return records


def count_instances(
    mcus: list[str],
    hdf_root: Path = HDF_ROOT,
) -> int:
    """Count total HDF instances for the given MCUs."""
    return sum(1 for _ in discover(hdf_root, mcus, (".hdf",)))

