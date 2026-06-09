import os
from pathlib import Path

import torch

from mcu_sample import McuSample


class McusDataset(torch.utils.data.Dataset):
    def __init__(
        self, mcus: list[str], data_path: Path, window_length: int, stats: dict
    ):
        self.samples: list[McuSample] = []
        self.window_map: list[tuple[int, int]] = []
        self.window_length = window_length
        self.stats = stats
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

        print(f"Mapped {len(self.window_map):,} windows from {len(mcus)} MCU(s).")

    def __len__(self):
        return len(self.window_map)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("dataset index out of range")

        sample_idx, start_idx = self.window_map[idx]
        sample = self.samples[sample_idx]
        end_idx = start_idx + self.window_length

        window = sample.load_window(start_idx, end_idx)

        raw_u = window[0, :]
        raw_i = window[1, :]
        raw_t = window[2, :]

        scaled_u = (raw_u - self.stats["U"]["mean"]) / self.stats["U"]["std"]
        scaled_i = (raw_i - self.stats["I"]["mean"]) / self.stats["I"]["std"]
        scaled_t = (raw_t - self.stats["Temp"]["mean"]) / self.stats["Temp"]["std"]

        u = torch.from_numpy(scaled_u).float()
        i = torch.from_numpy(scaled_i).float()
        t = torch.from_numpy(scaled_t).float()

        soh = torch.full((self.window_length,), sample.soh, dtype=torch.float32)

        condition_X = torch.stack([i, soh], dim=1)  # Perfect [1000, 2] Layout
        target_y = torch.stack([u, t], dim=1)  # Perfect [1000, 2] Layout

        return condition_X, target_y
