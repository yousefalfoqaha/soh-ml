import os
from pathlib import Path

import numpy as np

from mcu_sample import McuSample


def calculate_global_stats(data_path: Path, mcus: list[str]) -> dict:
    print("\nStarting global statistics calculation...")
    n_time_steps = 0

    u_sum = 0.0
    i_sum = 0.0
    t_sum = 0.0
    q_sum = 0.0
    qp_sum = 0.0
    ct_sum = 0.0

    u_sum_sq = 0.0
    i_sum_sq = 0.0
    t_sum_sq = 0.0
    q_sum_sq = 0.0
    qp_sum_sq = 0.0
    ct_sum_sq = 0.0

    for mcu in mcus:
        mcu_source_path = data_path / mcu

        if not mcu_source_path.exists():
            print(f"Source directory missing, skipping: {mcu_source_path}")
            continue

        for root, _, files in os.walk(mcu_source_path):
            for file in files:
                if file.lower().endswith(".hdf"):
                    sample_path = Path(root) / file
                    sample = McuSample(filepath=sample_path, qnom=18000)
                    n_samples = len(sample)
                    n_time_steps += n_samples

                    data = sample.load_window(start=0, end=n_samples)
                    u = data[0, :]
                    i = data[1, :]
                    t = data[2, :]
                    q = data[3, :]
                    qp = data[4, :]
                    ct = data[5, :]

                    u_sum += np.sum(u, dtype=np.float64)
                    i_sum += np.sum(i, dtype=np.float64)
                    t_sum += np.sum(t, dtype=np.float64)
                    q_sum += np.sum(q, dtype=np.float64)
                    qp_sum += np.sum(qp, dtype=np.float64)
                    ct_sum += np.sum(ct, dtype=np.float64)

                    u_sum_sq += np.sum(u**2, dtype=np.float64)
                    i_sum_sq += np.sum(i**2, dtype=np.float64)
                    t_sum_sq += np.sum(t**2, dtype=np.float64)
                    q_sum_sq += np.sum(q**2, dtype=np.float64)
                    qp_sum_sq += np.sum(qp**2, dtype=np.float64)
                    ct_sum_sq += np.sum(ct**2, dtype=np.float64)

    if n_time_steps == 0:
        raise ValueError("No data points found.")

    u_mean = u_sum / n_time_steps
    i_mean = i_sum / n_time_steps
    t_mean = t_sum / n_time_steps

    u_std = np.sqrt(max((u_sum_sq / n_time_steps) - (u_mean**2), 1e-8))
    i_std = np.sqrt(max((i_sum_sq / n_time_steps) - (i_mean**2), 1e-8))
    t_std = np.sqrt(max((t_sum_sq / n_time_steps) - (t_mean**2), 1e-8))
    q_mean = q_sum / n_time_steps
    q_std = np.sqrt(max((q_sum_sq / n_time_steps) - (q_mean**2), 1e-8))
    qp_mean = qp_sum / n_time_steps
    qp_std = np.sqrt(max((qp_sum_sq / n_time_steps) - (qp_mean**2), 1e-8))
    ct_mean = ct_sum / n_time_steps
    ct_std = np.sqrt(max((ct_sum_sq / n_time_steps) - (ct_mean**2), 1e-8))

    print("      GLOBAL SCALING STATISTICS CALCULATED")
    print(f"Total Time Steps Sampled : {n_time_steps:,}")
    print(f"Voltage (U)     -> Mean: {u_mean:10.4f} | Std: {u_std:10.4f}")
    print(f"Current (I)     -> Mean: {i_mean:10.4f} | Std: {i_std:10.4f}")
    print(f"Temp            -> Mean: {t_mean:10.4f} | Std: {t_std:10.4f}")
    print(f"Qneg            -> Mean: {q_mean:10.4f} | Std: {q_std:10.4f}")
    print(f"Qpos            -> Mean: {qp_mean:10.4f} | Std: {qp_std:10.4f}")
    print(f"ClimaTemp       -> Mean: {ct_mean:10.4f} | Std: {ct_std:10.4f}")

    return {
        "U": {"mean": float(u_mean), "std": float(u_std)},
        "I": {"mean": float(i_mean), "std": float(i_std)},
        "Temp": {"mean": float(t_mean), "std": float(t_std)},
        "Qneg": {"mean": float(q_mean), "std": float(q_std)},
        "Qpos": {"mean": float(qp_mean), "std": float(qp_std)},
        "ClimaTemp": {"mean": float(ct_mean), "std": float(ct_std)},
    }
