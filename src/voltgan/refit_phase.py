from datetime import datetime
from pathlib import Path

import h5py

from voltgan.config import AGING_END, AGING_START, HDF_ROOT


def _phase_of(dt: datetime) -> str:
    if dt < AGING_START:
        return "Initial"
    if dt <= AGING_END:
        return "Aging"
    return "Post-Aging"


def main() -> None:
    files = sorted(HDF_ROOT.rglob("*.hdf"))
    print(f"Scanning {len(files)} HDF files under {HDF_ROOT}...")

    n_updated = 0
    n_skipped = 0
    n_missing_dt = 0
    counts: dict[str, int] = {}

    for path in files:
        with h5py.File(path, "a") as f:
            if "phase" in f.attrs:
                existing = f.attrs["phase"]
                if isinstance(existing, str):
                    counts[existing] = counts.get(existing, 0) + 1
                    n_skipped += 1
                    continue
            dt_str = f.attrs.get("datetime")
            if not isinstance(dt_str, str):
                n_missing_dt += 1
                continue
            try:
                dt = datetime.fromisoformat(dt_str)
            except ValueError:
                n_missing_dt += 1
                continue
            phase = _phase_of(dt)
            f.attrs["phase"] = phase
            counts[phase] = counts.get(phase, 0) + 1
            n_updated += 1

    print(
        f"Updated: {n_updated} | Skipped (already had phase): {n_skipped} | "
        f"Missing datetime: {n_missing_dt}"
    )
    print("Phase distribution:")
    for phase, n in sorted(counts.items()):
        print(f"  {phase}: {n}")


if __name__ == "__main__":
    main()
