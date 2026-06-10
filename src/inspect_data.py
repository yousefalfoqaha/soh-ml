import os
from pathlib import Path

import numpy as np
from asammdf import MDF

DATA_PATH = Path(__file__).resolve().parent.parent / "data"
MF4_ROOT = DATA_PATH / "mf4"
MCUS = ["mcu1", "mcu2", "mcu3"]
CATEGORIES = ["initial", "aging", "after"]
CHANNELS = ["U", "I", "Temp[1]"]
N_INSPECT = 3

for mcu in MCUS:
    for cat in CATEGORIES:
        cat_path = MF4_ROOT / mcu / cat
        if not cat_path.exists():
            print(f"\n{'=' * 60}")
            print(f"Missing: {mcu}/{cat}")
            continue

        mf4_files = sorted(
            p
            for root, _, files in os.walk(cat_path)
            for f in files
            if f.lower().endswith(".mf4")
            for p in [Path(root) / f]
        )

        to_inspect = mf4_files[:N_INSPECT]

        print(f"\n{'=' * 60}")
        print(
            f"MCU: {mcu} | Category: {cat} | Files: {len(mf4_files)} total, inspecting {len(to_inspect)}"
        )
        print(f"{'=' * 60}")

        for fpath in to_inspect:
            mdf = MDF(fpath)

            print(f"\n  File: {fpath.name}")
            print(f"  Channels: {sorted(mdf.channels_db.keys())}")

            for ch in CHANNELS:
                if ch in mdf.channels_db:
                    sig = mdf.get(ch)
                    arr = np.asarray(sig.samples)
                    print(
                        f"  {ch:10s}: shape={arr.shape}  "
                        f"min={arr.min():.4f}  max={arr.max():.4f}  "
                        f"mean={arr.mean():.4f}  std={arr.std():.4f}"
                    )

            for key in sorted(mdf.channels_db.keys()):
                if key.startswith("sgl_") and key not in CHANNELS:
                    sig = mdf.get(key)
                    arr = np.asarray(sig.samples)
                    if arr.ndim == 0:
                        print(f"  META '{key}': {float(arr):.4f}")
                    elif arr.size <= 6:
                        print(f"  META '{key}': shape={arr.shape}  values={arr}")
                    else:
                        print(
                            f"  META '{key}': shape={arr.shape}  "
                            f"min={arr.min():.4f}  max={arr.max():.4f}  "
                            f"mean={arr.mean():.4f}"
                        )

            mdf.close()

    all_channels = []
    print(f"\n--- Channel coverage summary for {mcu} ---")
    for root, _, files in os.walk(MF4_ROOT / mcu):
        for f in files:
            if f.lower().endswith(".mf4"):
                try:
                    mdf = MDF(Path(root) / f)
                    all_channels.append((Path(root) / f, set(mdf.channels_db.keys())))
                    mdf.close()
                except Exception as e:
                    print(f"  ERROR loading {f}: {e}")

    if all_channels:
        all_keys = sorted(set().union(*(ch for _, ch in all_channels)))
        print(f"  Files: {len(all_channels)}")
        print(f"  All channels ({len(all_keys)}): {all_keys}")
        for key in all_keys:
            count = sum(1 for _, ch in all_channels if key in ch)
            print(f"    {key}: present in {count}/{len(all_channels)} files")
    else:
        print("  No MF4 files found.")

