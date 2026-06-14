import numpy as np
from asammdf import MDF


def _safe_get_channel(mdf: MDF, name: str) -> np.ndarray | None:
    channels = mdf.channels_db
    if name not in channels:
        return None
    try:
        sig = mdf.get(name)
        return sig.samples
    except Exception:
        return None


def _rasterized_current(
    mdf: MDF, raster: float
) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        sig = mdf.get("I", raster=raster)
        return sig.samples.astype(np.float64), sig.timestamps.astype(np.float64)
    except Exception:
        return None


def _rasterized_voltage(
    mdf: MDF, raster: float
) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        sig = mdf.get("U", raster=raster)
        return sig.samples.astype(np.float64), sig.timestamps.astype(np.float64)
    except Exception:
        return None


def _merge_intervals(
    starts: np.ndarray, ends: np.ndarray, gap: float
) -> list[tuple[float, float]]:
    intervals = sorted(zip(starts, ends), key=lambda x: x[0])
    merged: list[tuple[float, float]] = []
    for s, e in intervals:
        if merged and s - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((float(s), float(e)))
    return merged


def _contiguous_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    regions = []
    start = None
    for i in range(len(mask)):
        if mask[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                regions.append((start, i))
                start = None
    if start is not None:
        regions.append((start, len(mask)))
    return regions