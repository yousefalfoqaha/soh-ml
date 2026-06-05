from pathlib import Path

import h5py

TARGET_CHANNELS = ["U", "I", "Qneg", "Temp[1]"]


class McuSample:
    def __init__(self, filepath: Path, raster: float, qnom: int):
        hdf_path = filepath.with_suffix(".hdf")

        hdf = h5py.File(hdf_path)

        signal = mdf.get(name="U", raster=raster)
        qneg = mdf.get(name="Qneg", raster=raster)

        self.filepath = hdf_path
        self.n_samples = len(signal)
        self.soh = abs(float(qneg.samples.min())) / qnom * 100

    def __len__(self):
        return self.n_samples

    def load_window(self, start: int, end: int):
        print("WIP")
