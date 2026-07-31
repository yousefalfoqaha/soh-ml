from datetime import timedelta
from pathlib import Path

import numpy as np
import scipy.io as sio

from voltgan.config import (
    CHANNELS,
    EVALUATION_PROVIDER,
    OXFORD_AMBIENT_TEMPERATURE,
    OXFORD_BASE_DATETIME,
    OXFORD_NOMINAL_CAPACITY_MAH,
    OXFORD_PHASE,
    OXFORD_PROTOCOL,
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
            cell_key = f"Cell{cell_idx}"
            if cell_key not in mat:
                continue

            cell_data = mat[cell_key][0, 0]

            for cyc_name in cell_data.dtype.names:
                if not cyc_name.startswith("cyc"):
                    continue

                try:
                    dci = int(cyc_name[3:])
                except ValueError:
                    continue

                filename = f"Cyc{dci:03d}.hdf"
                if self.repo.exists(cell_key.lower(), filename):
                    continue

                try:
                    cyc_struct = cell_data[cyc_name][0, 0]
                    if "C1dc" not in cyc_struct.dtype.names:
                        continue
                    c1dc = cyc_struct["C1dc"][0, 0]

                    t = c1dc["t"].flatten()
                    v = c1dc["v"].flatten()
                    temp = c1dc["T"].flatten()
                except ValueError, IndexError, KeyError, TypeError:
                    continue

                if len(t) < self.min_seq_len:
                    continue

                i = np.full_like(t, -0.74, dtype=np.float32)

                capacity = float(abs(np.trapezoid(i, t)) / 3600.0)
                soh = min(capacity / (OXFORD_NOMINAL_CAPACITY_MAH / 1000.0), 1.0)

                self.repo.save(
                    cell_id=cell_key.lower(),
                    filename=filename,
                    data=dict(zip(CHANNELS, [v, i, temp])),
                    metadata={
                        "provider": EVALUATION_PROVIDER,
                        "cell_id": cell_key.lower(),
                        "soh": soh,
                        "curve_soh": soh,
                        "ambient_temperature": OXFORD_AMBIENT_TEMPERATURE,
                        "discharge_rate": 1.0,
                        "datetime": (
                            OXFORD_BASE_DATETIME + timedelta(days=dci)
                        ).isoformat(),
                        "discharge_cycle_index": dci,
                        "protocol": OXFORD_PROTOCOL,
                        "phase": OXFORD_PHASE,
                        "total_rows": len(t),
                    },
                )
