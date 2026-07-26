from __future__ import annotations

import re
from datetime import timedelta
from pathlib import Path
from typing import cast

import h5py
import numpy as np
from scipy.io import loadmat

from voltgan.config import (
    AMBIENT_TEMPERATURE_KEY,
    CURRENT_CHANNEL,
    MAX_SEQUENCE_LENGTH,
    MIN_SEQUENCE_LENGTH,
    OXFORD_AMBIENT_TEMPERATURE,
    OXFORD_BASE_DATETIME,
    OXFORD_DISCHARGE_CURRENT_MA,
    OXFORD_FINE_TUNE_FRACTION,
    OXFORD_HDF_DIR,
    OXFORD_MAT_PATH,
    OXFORD_N_CELLS,
    OXFORD_NOMINAL_CAPACITY_MAH,
    OXFORD_PHASE,
    OXFORD_PROTOCOL,
    RASTER_FREQUENCY,
    TEMPERATURE_CHANNEL,
    VOLTAGE_CHANNEL,
)

_MATLAB_DATENUM_DAY = 86400.0
_COLD_START_MARGIN = 5.0
_CYC_PATTERN = re.compile(r"^cyc(\d+)$")


def _extract_c1dc(cycle: dict) -> dict[str, np.ndarray]:
    c1dc = cycle["C1dc"]
    return {
        "t": np.asarray(c1dc["t"], dtype=np.float64).ravel(),
        "v": np.asarray(c1dc["v"], dtype=np.float64).ravel(),
        "q": np.asarray(c1dc["q"], dtype=np.float64).ravel(),
        "T": np.asarray(c1dc["T"], dtype=np.float64).ravel(),
    }


def _trim_cold_start(
    signals: dict[str, np.ndarray], ambient: float
) -> dict[str, np.ndarray]:
    mask = signals["T"] >= ambient - _COLD_START_MARGIN
    if not np.any(mask):
        return signals
    first = int(np.argmax(mask))
    if first == 0:
        return signals
    return {k: v[first:] for k, v in signals.items()}


def _resample(signals: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    t_days = signals["t"]
    t0 = t_days[0]
    t_seconds = (t_days - t0) * _MATLAB_DATENUM_DAY
    duration = float(t_seconds[-1])
    new_index = np.arange(0.0, duration + RASTER_FREQUENCY, RASTER_FREQUENCY)
    new_index = new_index[new_index <= duration]
    if new_index.size == 0:
        return {}

    resampled = {}
    for channel in ("v", "q", "T"):
        resampled[channel] = np.interp(new_index, t_seconds, signals[channel]).astype(
            np.float64
        )
    return resampled


def _enrich_file(file_path: Path) -> None:
    with h5py.File(file_path, "a") as f:
        group = cast(h5py.Group, f[file_path.name])

        total_rows = 0
        for channel in group.keys():
            dataset = group[channel]
            if not isinstance(dataset, h5py.Dataset):
                continue
            total_rows = len(dataset)
            data = dataset[:]
            f.attrs[f"{channel}_mean"] = float(np.mean(data))
            f.attrs[f"{channel}_m2"] = float(np.var(data) * total_rows)
            f.attrs[f"{channel}_min"] = float(np.min(data))
            f.attrs[f"{channel}_max"] = float(np.max(data))

        f.attrs["total_rows"] = total_rows
        f.attrs[f"{AMBIENT_TEMPERATURE_KEY}_mean"] = f.attrs[AMBIENT_TEMPERATURE_KEY]
        f.attrs[f"{AMBIENT_TEMPERATURE_KEY}_m2"] = 0.0
        f.attrs[f"{AMBIENT_TEMPERATURE_KEY}_min"] = f.attrs[AMBIENT_TEMPERATURE_KEY]
        f.attrs[f"{AMBIENT_TEMPERATURE_KEY}_max"] = f.attrs[AMBIENT_TEMPERATURE_KEY]


def _write_hdf(
    target: Path,
    voltage: np.ndarray,
    current: np.ndarray,
    temperature: np.ndarray,
    soh: float,
    ambient_temperature: float,
    discharge_cycle_index: int,
    datetime_iso: str,
    protocol: str,
    phase: str,
    split: str,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(target, "w") as f:
        group = f.create_group(target.name)
        group.create_dataset(VOLTAGE_CHANNEL, data=voltage)
        group.create_dataset(CURRENT_CHANNEL, data=current)
        group.create_dataset(TEMPERATURE_CHANNEL, data=temperature)
        f.attrs["soh"] = soh
        f.attrs["curve_soh"] = soh
        f.attrs[AMBIENT_TEMPERATURE_KEY] = ambient_temperature
        f.attrs["mean_neg_current"] = abs(OXFORD_DISCHARGE_CURRENT_MA) / 1000.0
        f.attrs["datetime"] = datetime_iso
        f.attrs["discharge_cycle_index"] = discharge_cycle_index
        f.attrs["protocol"] = protocol
        f.attrs["phase"] = phase
        f.attrs["split"] = split

    _enrich_file(target)


def _process_cell(cell: dict, cell_index: int, hdf_dir: Path) -> int:
    cyc_keys = sorted(k for k in cell.keys() if _CYC_PATTERN.match(k))
    if not cyc_keys:
        print(f"  Cell{cell_index}: no cycNNNN keys, skipping")
        return 0

    cycle_indices = []
    for k in cyc_keys:
        m = _CYC_PATTERN.match(k)
        assert m is not None
        cycle_indices.append(int(m.group(1)))
    max_cycle_index = max(cycle_indices) if cycle_indices else 0
    written = 0
    skipped_short = 0

    for cyc_key, cycle_index in zip(cyc_keys, cycle_indices):
        cycle = cell[cyc_key]
        if "C1dc" not in cycle:
            continue

        raw = _extract_c1dc(cycle)
        if raw["v"].size == 0:
            continue

        trimmed = _trim_cold_start(raw, OXFORD_AMBIENT_TEMPERATURE)
        if trimmed["v"].size < 2:
            continue

        resampled = _resample(trimmed)
        if not resampled:
            skipped_short += 1
            continue

        voltage = resampled["v"][:MAX_SEQUENCE_LENGTH]
        temperature = resampled["T"][:MAX_SEQUENCE_LENGTH]
        n_samples = voltage.size
        current = np.full(
            n_samples, -OXFORD_DISCHARGE_CURRENT_MA / 1000.0, dtype=np.float64
        )

        if n_samples < MIN_SEQUENCE_LENGTH:
            skipped_short += 1
            continue

        soh = min(float(np.abs(trimmed["q"]).max()) / OXFORD_NOMINAL_CAPACITY_MAH, 1.0)

        split = (
            "train"
            if cycle_index < max_cycle_index * OXFORD_FINE_TUNE_FRACTION
            else "test"
        )
        datetime_iso = (
            OXFORD_BASE_DATETIME + timedelta(hours=cycle_index * 100)
        ).isoformat()

        filename = f"oxford_cell{cell_index}_cyc{cycle_index:04d}.hdf"
        target = hdf_dir / filename
        _write_hdf(
            target,
            voltage=voltage,
            current=current,
            temperature=temperature,
            soh=soh,
            ambient_temperature=OXFORD_AMBIENT_TEMPERATURE,
            discharge_cycle_index=cycle_index,
            datetime_iso=datetime_iso,
            protocol=OXFORD_PROTOCOL,
            phase=OXFORD_PHASE,
            split=split,
        )
        written += 1

    print(f"  Cell{cell_index}: {written} cycles written ({skipped_short} skipped)")
    return written


def main() -> None:
    print("Starting Oxford preprocessing pipeline...")
    if not OXFORD_MAT_PATH.exists():
        raise FileNotFoundError(f"Oxford .mat not found at {OXFORD_MAT_PATH}")

    data = loadmat(OXFORD_MAT_PATH, simplify_cells=True)
    OXFORD_HDF_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    for cell_index in range(1, OXFORD_N_CELLS + 1):
        cell = data[f"Cell{cell_index}"]
        total += _process_cell(cell, cell_index, OXFORD_HDF_DIR)
    print(f"Total Oxford cycles written: {total}")

    print("Oxford preprocessing complete.")


if __name__ == "__main__":
    main()

