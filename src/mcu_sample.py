from pathlib import Path


class McuSample:
    def __init__(self, filepath: Path, length: int):
        self.filepath = filepath
        self.length = length

    def __len__(self):
        return self.length
