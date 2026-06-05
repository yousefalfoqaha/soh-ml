import os
from pathlib import Path

import torch

from mcu_sample import McuSample


class McusDataset(torch.utils.data.Dataset):
    def __init__(self, mcus: list[str], data_path: Path, window_length: int):
        self.samples: list[McuSample] = []
        self.window_map: list[tuple[int, int]] = []
        self.window_length = window_length
        stride = window_length

        for mcu in mcus:
            for root, _, files in os.walk(data_path / mcu):
                for file in files:
                    path = Path(root) / file
                    sample = McuSample(filepath=path, qnom=18000)
                    sample_idx = len(self.samples)
                    self.samples.append(sample)
                    n_windows = (sample.n_samples - window_length) // stride + 1

                    for i in range(n_windows):
                        start_idx = i * stride
                        self.window_map.append((sample_idx, start_idx))

    def __len__(self):
        return len(self.window_map)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("dataset index out of range")

        sample_idx, start_idx = self.window_map[idx]
        sample = self.samples[sample_idx]
        end_idx = start_idx + self.window_length

        window = sample.load_window(start_idx, end_idx)

        u = torch.from_numpy(window[0:1, :]).float()
        i = torch.from_numpy(window[1:2, :]).float()
        t = torch.from_numpy(window[2:3, :]).float()

        soh = torch.full((1, self.window_length), sample.soh, dtype=torch.float32)

        condition_X = torch.cat([i, soh], dim=0)
        target_y = torch.cat([u, t], dim=0)

        return condition_X, target_y
