import h5py
import numpy as np
from pathlib import Path

from voltgan.config import DATA_PATH, TRAINING_MCUS, TESTING_MCUS, VALIDATION_MCUS
from voltgan.pipeline.stats_enricher import StatsEnrichHandler


def main():
    handler = StatsEnrichHandler()
    hdf_root = DATA_PATH / "hdf"
    all_mcus = TRAINING_MCUS + VALIDATION_MCUS + TESTING_MCUS

    for mcu in all_mcus:
        mcu_path = hdf_root / mcu
        if not mcu_path.exists():
            print(f"Skipping {mcu}: directory not found")
            continue

        hdf_files = sorted(mcu_path.rglob("*.hdf"))
        print(f"Re-enriching {mcu}: {len(hdf_files)} files")

        for i, fp in enumerate(hdf_files):
            with h5py.File(fp, "a") as f:
                group = f[fp.name]
                assert isinstance(group, h5py.Group)

                for channel in list(group.keys()):
                    dataset = group[channel]
                    if not isinstance(dataset, h5py.Dataset):
                        continue
                    data = dataset[:]
                    if f"{channel}_min" not in f.attrs:
                        f.attrs[f"{channel}_min"] = float(np.min(data))
                    if f"{channel}_max" not in f.attrs:
                        f.attrs[f"{channel}_max"] = float(np.max(data))

                if "ambient_temperature_min" not in f.attrs:
                    f.attrs["ambient_temperature_min"] = f.attrs["ambient_temperature"]
                if "ambient_temperature_max" not in f.attrs:
                    f.attrs["ambient_temperature_max"] = f.attrs["ambient_temperature"]

            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(hdf_files)}")

    print("Re-enrichment complete.")


if __name__ == "__main__":
    main()