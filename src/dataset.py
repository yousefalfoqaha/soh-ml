import os
from pathlib import Path

import torch
from asammdf import MDF

from mcu_sample import McuSample


class McusDataset(torch.utils.data.Dataset):
    def __init__(self, mcus, data_path, window_length):
        samples = []

        for mcu in mcus:
            mcu_path = data_path / mcu

            for root, _, files in os.walk(mcu_path):
                for file in files:
                    sample_path = Path(root) / file
                    sample_mdf = MDF(sample_path)

        self.samples = samples
        self.window_length = window_length

    def __len__(self):
        return len(self.series) - self.window_length

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("dataset index out of range")
        end = idx + self.window_length
        window = self.series[idx:end]
        target = self.series[end]
        return window, target
