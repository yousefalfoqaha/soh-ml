import os
from pathlib import Path

import h5py
import numpy as np

from mcu_sample import McuSample

DATA_PATH = Path("../data")
HDF_ROOT = DATA_PATH / "hdf"
MCUS = ["mcu1", "mcu2", "mcu3"]
CATEGORIES = ["initial", "aging", "after"]
N_INSPECT = 3

for mcu in MCUS:
    for cat in CATEGORIES:
        cat_path = HDF_ROOT / mcu / cat
        if not cat_path.exists():
            print(f"\n{'=' * 60}")
            print(f"Missing: {mcu}/{cat}")
            continue

        hdf_files = sorted(
            p
            for root, _, files in os.walk(cat_path)
            for f in files
            if f.lower().endswith(".hdf")
            for p in [Path(root) / f]
        )

        to_inspect = hdf_files[:N_INSPECT]

        print(f"\n{'=' * 60}")
        print(f"MCU: {mcu} | Category: {cat} | Files: {len(hdf_files)} total, inspecting {len(to_inspect)}")
        print(f"{'=' * 60}")

        for hpath in to_inspect:
            sample = McuSample(filepath=hpath, qnom=18000)

            print(f"\n  File: {hpath.name}")
            print(f"  n_samples : {sample.n_samples:,}")
            print(f"  SoH       : {sample.soh:.4f}")

            data = sample.load_window(0, sample.n_samples)
            print(
                f"  U (row0)  : min={data[0].min():.4f}  max={data[0].max():.4f}  "
                f"mean={data[0].mean():.4f}  std={data[0].std():.4f}"
            )
            print(
                f"  I (row1)  : min={data[1].min():.4f}  max={data[1].max():.4f}  "
                f"mean={data[1].mean():.4f}  std={data[1].std():.4f}"
            )
            print(
                f"  T (row2)  : min={data[2].min():.4f}  max={data[2].max():.4f}  "
                f"mean={data[2].mean():.4f}  std={data[2].std():.4f}"
            )

            with h5py.File(hpath, "r") as f:
                group = f[sample._group_path]
                for key in sorted(group.keys()):
                    ds = group[key]
                    if isinstance(ds, h5py.Dataset):
                        arr = ds[:]
                        print(
                            f"  HDF '{key}'  : shape={ds.shape}  "
                            f"min={arr.min():.4f}  max={arr.max():.4f}  "
                            f"mean={arr.mean():.4f}  std={arr.std():.4f}"
                        )

    all_soh = []
    print(f"\n--- SoH summary for {mcu} ---")
    for root, _, files in os.walk(HDF_ROOT / mcu):
        for f in files:
            if f.lower().endswith(".hdf"):
                try:
                    s = McuSample(filepath=Path(root) / f, qnom=18000)
                    all_soh.append((Path(root) / f, s.soh))
                except Exception as e:
                    print(f"  ERROR loading {f}: {e}")

    if all_soh:
        soh_vals = [s for _, s in all_soh]
        print(
            f"  Files: {len(all_soh)} | "
            f"SoH min={min(soh_vals):.4f} max={max(soh_vals):.4f} "
            f"mean={np.mean(soh_vals):.4f} std={np.std(soh_vals):.4f}"
        )
        outliers = [(p, s) for p, s in all_soh if s < 0 or s > 100]
        if outliers:
            print("  OUTLIER SoH values (outside 0-100):")
            for p, s in outliers:
                print(f"    {p.relative_to(HDF_ROOT)}: SoH={s:.4f}")
    else:
        print("  No HDF files found.")