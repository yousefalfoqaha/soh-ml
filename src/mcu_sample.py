from pathlib import Path

from asammdf import MDF


class McuSample:
    def __init__(self, filepath: Path, raster_freq: float, qnom: int):
        self.filepath = filepath
        mdf = MDF(filepath)
        signal = mdf.get(name="U", raster=raster_freq)
        qneg = mdf.get(name="Qneg", raster=raster_freq)

        self.n_samples = len(signal)
        self.path = filepath
        self.soh = abs(float(qneg.samples.min())) / qnom * 100

        mdf.close()

    def __len__(self):
        return self.n_samples
