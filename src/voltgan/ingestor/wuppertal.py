import math
from pathlib import Path

import numpy as np
from asammdf import MDF

from voltgan.config import (
    AGING_END,
    AGING_START,
    CHANNELS,
    CURRENT_CHANNEL,
    NOMINAL_CAPACITY,
    TESTING_MCUS,
    TRAINING_MCUS,
    VALIDATION_MCUS,
    WUPPERTAL_PROVIDER,
)
from voltgan.dataset.repository import InstanceRepository
from voltgan.ingestor.base import DatasetIngestor, Window
from voltgan.utils.discover import FileDiscoverer


class WuppertalIngestor(DatasetIngestor):
    def __init__(
        self,
        mf4_dir: Path,
        raster: float,
        min_seq_len: int,
        repo: InstanceRepository,
    ):
        self.raw_dir = mf4_dir
        self._mcus = TRAINING_MCUS + VALIDATION_MCUS + TESTING_MCUS
        self.raster = raster
        self.min_seq_len = min_seq_len
        self.repo = repo
        self.amb_temp_ch = "ClimaTemp"

    def ingest(self) -> None:
        files = FileDiscoverer.find(self.raw_dir, self._mcus, (".mf4",))

        for mcu in self._mcus:
            dci = 0
            mcu_files = [
                f for f in files if f.relative_to(self.raw_dir).parts[0] == mcu
            ]

            for mf4_path in mcu_files:
                dci = self._process_file(mf4_path, dci, mcu)

    def _process_file(self, mf4_path: Path, dci: int, mcu: str) -> int:
        req_channels = CHANNELS + [self.amb_temp_ch]
        time_channels = [
            "sgl_charge_time_start",
            "sgl_discharge_time_start",
            "sgl_discharge_time_end",
            "sgl_pulse",
        ]

        mdf = MDF(name=mf4_path, channels=req_channels + time_channels)
        if any(ch not in mdf.channels_db for ch in req_channels):
            mdf.close()
            return dci

        windows = self._calculate_soh_and_ambient(mdf, self._extract_periods(mdf))
        dci = self._export_hdf(mdf, mf4_path, windows, dci, mcu)
        mdf.close()
        return dci

    def _extract_periods(self, mdf: MDF) -> list[Window]:
        db = mdf.channels_db
        d_start = (
            mdf.get("sgl_discharge_time_start").samples.astype(np.float32)
            if "sgl_discharge_time_start" in db
            else np.array([])
        )
        d_end = (
            mdf.get("sgl_discharge_time_end").samples.astype(np.float32)
            if "sgl_discharge_time_end" in db
            else np.array([])
        )
        c_start = (
            mdf.get("sgl_charge_time_start").samples.astype(np.float32)
            if "sgl_charge_time_start" in db
            else np.array([])
        )
        p_sig = mdf.get("sgl_pulse") if "sgl_pulse" in db else None

        if not len(c_start) and not (len(d_start) or p_sig):
            return [Window(-np.inf, np.inf, "WLTC")]
        if len(c_start) and not (len(d_start) or p_sig):
            return []

        search = (
            [(-np.inf, np.inf)]
            if not len(c_start) or np.all(c_start == c_start.mean())
            else list(zip(c_start[:-1], c_start[1:]))
        )

        instances = []
        for w_s, w_e in search:
            mask = (d_start >= w_s) & (d_start < w_e)
            if np.sum(mask) > 0:
                instances.append(
                    Window(
                        float(d_start[mask].min()),
                        float(d_end[mask].max()),
                        "Constant" if np.sum(mask) == 1 else "HPPC",
                    )
                )
            elif p_sig:
                p_mask = (p_sig.timestamps >= w_s) & (p_sig.timestamps < w_e)
                if np.any(p_mask):
                    instances.append(
                        Window(
                            float(p_sig.timestamps[p_mask].min()),
                            float(p_sig.timestamps[p_mask].max()),
                            "Pulse",
                        )
                    )
        return instances

    def _calculate_soh_and_ambient(
        self, mdf: MDF, windows: list[Window]
    ) -> list[Window]:
        i_samps, i_times = (
            mdf.get(CURRENT_CHANNEL).samples,
            mdf.get(CURRENT_CHANNEL).timestamps,
        )
        t_samps, t_times = (
            mdf.get(self.amb_temp_ch).samples,
            mdf.get(self.amb_temp_ch).timestamps,
        )

        valid = []
        for w in windows:
            i_mask = (i_times >= w.start) & (i_times <= w.end)
            if not np.any(i_mask):
                continue

            cur = i_samps[i_mask]
            w.soh = min(
                abs(float(np.trapezoid(cur, i_times[i_mask]))) / NOMINAL_CAPACITY,
                1.0,
            )

            t_mask = (t_times >= w.start) & (t_times <= w.end)
            w.amb = (
                float(t_samps[t_mask][np.sum(t_mask) // 2])
                if np.any(t_mask)
                else float("nan")
            )

            if w.protocol in ("Constant", "Pulse"):
                min_current = float(np.min(cur))
                capacity_ah = NOMINAL_CAPACITY / 3600.0
                raw_rate = abs(min_current) / capacity_ah
                w.discharge_rate = round(raw_rate, 1)
            else:
                w.discharge_rate = None

            valid.append(w)
        return valid

    def _export_hdf(
        self,
        mdf: MDF,
        source_path: Path,
        windows: list[Window],
        cycle_index: int,
        mcu: str,
    ) -> int:
        if not windows:
            return cycle_index

        df = mdf.to_dataframe(channels=CHANNELS, raster=None, time_from_zero=False)
        dt = FileDiscoverer.parse_datetime(source_path)
        phase = (
            "Initial"
            if dt < AGING_START
            else ("Aging" if dt <= AGING_END else "Post-Aging")
        )

        dt_str = dt.strftime("%Y%m%d")

        for w in windows:
            temp_str = (
                "TempNaN" if math.isnan(w.amb) else f"Temp{int(round(w.amb / 5) * 5)}"
            )

            name_parts = [f"Cyc{cycle_index:03d}", phase, w.protocol]

            if w.discharge_rate is not None:
                name_parts.append(f"{w.discharge_rate}C")

            name_parts.append(temp_str)
            name_parts.append(dt_str)

            filename = "_".join(name_parts) + ".hdf"

            if self.repo.exists(cell_id=mcu, filename=filename):
                cycle_index += 1
                continue

            resampled, new_idx = self._resample_window(df, w)
            if resampled is None or new_idx is None:
                continue

            self.repo.save(
                cell_id=mcu,
                filename=filename,
                data=resampled,
                metadata={
                    "provider": WUPPERTAL_PROVIDER,
                    "cell_id": mcu,
                    "soh": w.soh,
                    "ambient_temperature": w.amb,
                    "datetime": dt.isoformat(),
                    "discharge_cycle_index": cycle_index,
                    "protocol": w.protocol,
                    "phase": phase,
                    "discharge_rate": w.discharge_rate,
                    "total_rows": len(new_idx),
                },
            )

            cycle_index += 1

        return cycle_index

    def _resample_window(self, df, w: Window):
        start_t, end_t = max(w.start, df.index[0]), min(w.end, df.index[-1])
        idf = df.loc[start_t:end_t]
        orig_idx = idf.index.to_numpy() - start_t
        if not len(orig_idx):
            return None, None

        new_idx = np.arange(0, float(orig_idx[-1]) + self.raster, self.raster)
        new_idx = new_idx[new_idx <= float(orig_idx[-1])]
        if len(new_idx) < self.min_seq_len:
            return None, None

        resampled = {
            ch: np.interp(new_idx, orig_idx, idf[ch].to_numpy()) for ch in idf.columns
        }
        return resampled, new_idx
