from pathlib import Path

from asammdf import MDF

TARGET_CHANNELS = ["U", "I", "Qneg", "Temp[1]"]


class McuSample:
    def __init__(self, filepath: Path, raster: float, qnom: int):
        mdf = MDF(name=filepath, channels=TARGET_CHANNELS)

        hdf5_path = filepath.with_suffix(".hd5")
        if not hdf5_path.exists():
            mdf.export(
                fmt="hdf5",
                filename=hdf5_path,
                single_time_base=True,
                raster=raster,
            )

        mdf.close()

        signal = mdf.get(name="U", raster=raster)
        qneg = mdf.get(name="Qneg", raster=raster)

        self.filepath = hdf5_path
        self.n_samples = len(signal)
        self.soh = abs(float(qneg.samples.min())) / qnom * 100

    def __len__(self):
        return self.n_samples

    def load_window(self, start: int, end: int):
        print("WIP")
