from pathlib import Path

import torch

from voltgan.data.mcu_sample import McuSample
from voltgan.pipeline.base import discover


class McusDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        mcus: list[str],
        data_path: Path,
        window_length: int,
        stride: int,
        stats: dict,
    ):
        self.samples: list[McuSample] = []
        self.window_map: list[tuple[int, int]] = []
        self.window_length = window_length
        self.stats = stats
        self.stride = stride

        for hdf_path in discover(data_path, mcus, (".hdf",)):
            sample = McuSample(filepath=hdf_path)
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
        raw_ct = window[3, :]

        scaled_u = (raw_u - self.stats["U"]["mean"]) / self.stats["U"]["std"]
        scaled_i = (raw_i - self.stats["I"]["mean"]) / self.stats["I"]["std"]
        scaled_t = (raw_t - self.stats["Temp[1]"]["mean"]) / self.stats["Temp[1]"][
            "std"
        ]
        scaled_ct = (raw_ct - self.stats["ClimaTemp"]["mean"]) / self.stats[
            "ClimaTemp"
        ]["std"]

        u = torch.from_numpy(scaled_u).float()
        i = torch.from_numpy(scaled_i).float()
        t = torch.from_numpy(scaled_t).float()
        ct = torch.from_numpy(scaled_ct).float()

        condition_X = torch.stack([i, ct], dim=1)
        init_condition = torch.tensor(
            [scaled_u[0], scaled_t[0], sample.soh], dtype=torch.float32
        )
        target_y = torch.stack([u, t], dim=1)

        return condition_X, init_condition, target_y
