from datetime import timedelta
from pathlib import Path

import numpy as np
import scipy.io as sio

from voltgan.config import (
    CHANNELS,
    OXFORD_AMBIENT_TEMPERATURE,
    OXFORD_BASE_DATETIME,
    OXFORD_NOMINAL_CAPACITY_MAH,
    OXFORD_PHASE,
    OXFORD_PROTOCOL,
    OXFORD_PROVIDER,
)
from voltgan.dataset.repository import InstanceRepository
from voltgan.ingestor.base import DatasetIngestor


class OxfordIngestor(DatasetIngestor):
    def __init__(self, mat_path: Path, min_seq_len: int, repo: InstanceRepository):
        self.mat_path = mat_path
        self.min_seq_len = min_seq_len
        self.repo = repo

    def ingest(self) -> None:
        if not self.mat_path.exists():
            return

        mat = sio.loadmat(self.mat_path)
        for cell_idx in range(1, 9):
            cell_key = f"cell{cell_idx}"
            if cell_key not in mat:
                continue

            for dci, cycle in enumerate(mat[cell_key][0, 0]["cyc"][0]):
                filename = f"cycle-{dci:04d}.hdf"

                if self.repo.exists(cell_key, filename):
                    continue

                t = cycle["t"][0, 0].flatten()
                v = cycle["v"][0, 0].flatten()
                i = cycle["I"][0, 0].flatten()
                temp = cycle["T"][0, 0].flatten()

                if len(t) < self.min_seq_len:
                    continue

                capacity = float(abs(np.trapezoid(i, t)) / 3600.0)
                soh = min(capacity / (OXFORD_NOMINAL_CAPACITY_MAH / 1000.0), 1.0)

                self.repo.save(
                    cell_id=cell_key,
                    filename=filename,
                    data=dict(zip(CHANNELS, [v, i, temp])),
                    metadata={
                        "provider": OXFORD_PROVIDER,
                        "cell_id": cell_key,
                        "soh": soh,
                        "curve_soh": soh,
                        "ambient_temperature": OXFORD_AMBIENT_TEMPERATURE,
                        "discharge_rate": 1.0,
                        "mean_neg_current": float(np.mean(np.abs(i[i < 0])))
                        if np.any(i < 0)
                        else 0.0,
                        "datetime": (
                            OXFORD_BASE_DATETIME + timedelta(days=dci)
                        ).isoformat(),
                        "discharge_cycle_index": dci,
                        "protocol": OXFORD_PROTOCOL,
                        "phase": OXFORD_PHASE,
                        "total_rows": len(t),
                    },
                )
