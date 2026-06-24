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
            sample_i = len(self.samples)
            self.samples.append(sample)
            n_windows = (sample.n_samples - window_length) // stride + 1

            for i in range(n_windows):
                start_i = i * stride
                self.window_map.append((sample_i, start_i))

        print(f"Mapped {len(self.window_map):,} windows from {len(mcus)} MCU(s).")

    def __len__(self):
        return len(self.window_map)

    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError("dataset index out of range")

        sample_i, start_i = self.window_map[i]
        sample = self.samples[sample_i]
        end_i = start_i + self.window_length

        window = sample.load_window(start_i, end_i)

        voltage_scaled = (window[0, :] - self.stats["U"]["mean"]) / self.stats["U"][
            "std"
        ]
        current_scaled = (window[1, :] - self.stats["I"]["mean"]) / self.stats["I"][
            "std"
        ]
        temperature_scaled = (
            window[2, :] - self.stats["Temp[1]"]["mean"]
        ) / self.stats["Temp[1]"]["std"]
        ambient_temperature_scaled = (
            window[3, :] - self.stats["ClimaTemp"]["mean"]
        ) / self.stats["ClimaTemp"]["std"]

        voltage = torch.from_numpy(voltage_scaled).float()
        current = torch.from_numpy(current_scaled).float()
        temperature = torch.from_numpy(temperature_scaled).float()
        ambient_temperature = torch.from_numpy(ambient_temperature_scaled).float()

        # (window_length, 2)
        X = torch.stack([current, ambient_temperature], dim=1)

        # (3,)
        initial_conditions = torch.tensor(
            [voltage_scaled[0], temperature_scaled[0], sample.soh], dtype=torch.float32
        )

        # (window_length, 2)
        y = torch.stack([voltage, temperature], dim=1)

        return X, initial_conditions, y
