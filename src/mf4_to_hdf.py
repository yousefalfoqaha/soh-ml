import os
from pathlib import Path

from asammdf import MDF


def convert_to_hdf(
    data_path: Path, mcus: list[str], raster: float, channels: list[str]
):
    mf4_root = data_path / "mf4"
    hdf_root = data_path / "hdf"

    for mcu in mcus:
        mcu_source_path = mf4_root / mcu

        if not mcu_source_path.exists():
            print(f"Source directory missing, skipping: {mcu_source_path}")
            continue

        for root, _, files in os.walk(mcu_source_path):
            for file in files:
                if file.lower().endswith((".dat", ".mf4")):
                    mf4_path = Path(root) / file

                    relative_mf4_path = mf4_path.relative_to(mf4_root)
                    hdf_path = (hdf_root / relative_mf4_path).with_suffix(".hdf")

                    if not hdf_path.exists():
                        hdf_path.parent.mkdir(parents=True, exist_ok=True)

                        print(
                            f"Converting {mf4_path.name} -> {hdf_path.relative_to(hdf_root)}..."
                        )
                        mdf = MDF(name=mf4_path, channels=channels)
                        mdf.export(
                            fmt="hdf5",
                            filename=hdf_path.name,
                            single_time_base=True,
                            raster=raster,
                        )
                        mdf.close()

                        os.rename(hdf_path.name, hdf_path)
                        print(f"Finished converting: {hdf_path.name}")
