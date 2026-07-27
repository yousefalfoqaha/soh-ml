from pathlib import Path

import numpy as np
from asammdf import MDF

from voltgan.config import AGING_END, AGING_START, CHANNELS, CURRENT_CHANNEL
from voltgan.dataset.repository import InstanceRepository
from voltgan.dataset.soh_curve import SohCurveFitter
from voltgan.ingestor.base import DatasetIngestor, Window
from voltgan.utils.discover import FileDiscoverer


class WuppertalIngestor(DatasetIngestor):
    def __init__(
        self,
        raw_dir: Path,
        mcus: list[str],
        nominal_capacity: float,
        raster: float,
        min_seq_len: int,
        repo: InstanceRepository,
        fitter: SohCurveFitter,
    ):
        self.raw_dir = raw_dir
        self.mcus = mcus
        self.nominal_capacity = nominal_capacity
        self.raster = raster
        self.min_seq_len = min_seq_len
        self.repo = repo
        self.fitter = fitter
        self.amb_temp_ch = "ClimaTemp"

    def ingest(self) -> None:
        files = FileDiscoverer.find(self.raw_dir, self.mcus, (".mf4", ".dat"))
        files.sort(key=FileDiscoverer.sort_wuppertal)

        for mcu in self.mcus:
            dci = 0
            mcu_files = [
                f for f in files if f.relative_to(self.raw_dir).parts[0] == mcu
            ]

            for mf4_path in mcu_files:
                dci = self._process_file(mf4_path, dci, mcu)

            self._apply_soh_curve(mcu)

    def _apply_soh_curve(self, mcu: str) -> None:
        instances = self.repo.load([mcu])
        if not instances:
            return

        records = [
            (inst.dci, inst.soh, inst.ambient_temperature, inst.mean_neg_current)
            for inst in instances
        ]

        fit_result = self.fitter.fit(records)
        if not fit_result:
            print(f"[{mcu}] Skipped curve fitting - insufficient reference points.")
            return

        print(
            f"[{mcu}] Fitted deg4 curve | {len(fit_result.ref_points)} pts | RMSE: {fit_result.rmse:.5f}"
        )

        for inst in instances:
            fitted_soh = min(max(float(fit_result.model(inst.dci)), 0.0), 1.0)
            self.repo.update_metadata(inst.filepath, {"curve_soh": fitted_soh})

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

        windows = self._calc_soh_and_amb(mdf, self._extract_periods(mdf))
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

    def _calc_soh_and_amb(self, mdf: MDF, windows: list[Window]) -> list[Window]:
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
                abs(float(np.trapezoid(cur, i_times[i_mask]))) / self.nominal_capacity,
                1.0,
            )
            w.mnc = float(np.mean(np.abs(cur[cur < 0]))) if np.any(cur < 0) else 0.0

            t_mask = (t_times >= w.start) & (t_times <= w.end)
            w.amb = (
                float(t_samps[t_mask][np.sum(t_mask) // 2])
                if np.any(t_mask)
                else float("nan")
            )
            valid.append(w)
        return valid

    def _export_hdf(
        self, mdf: MDF, source_path: Path, windows: list[Window], dci: int, mcu: str
    ) -> int:
        if not windows:
            return dci

        base_path = self.repo.root / source_path.relative_to(self.raw_dir).with_suffix(
            ""
        )
        df = mdf.to_dataframe(channels=CHANNELS, raster=None, time_from_zero=False)
        dt = FileDiscoverer.sort_wuppertal(source_path)
        phase = (
            "Initial"
            if dt < AGING_START
            else ("Aging" if dt <= AGING_END else "Post-Aging")
        )

        for i, w in enumerate(windows):
            out_file = base_path.parent / (
                f"{base_path.name}.hdf"
                if len(windows) == 1
                else f"{base_path.name}_{i + 1}.hdf"
            )
            if out_file.exists():
                dci += 1
                continue

            start_t, end_t = max(w.start, df.index[0]), min(w.end, df.index[-1])
            idf = df.loc[start_t:end_t]
            orig_idx = idf.index.to_numpy() - start_t
            if not len(orig_idx):
                continue

            new_idx = np.arange(0, float(orig_idx[-1]) + self.raster, self.raster)
            new_idx = new_idx[new_idx <= float(orig_idx[-1])]
            if len(new_idx) < self.min_seq_len:
                continue

            resampled = {
                ch: np.interp(new_idx, orig_idx, idf[ch].to_numpy())
                for ch in idf.columns
            }

            self.repo.save(
                filepath=out_file,
                data=resampled,
                metadata={
                    "cell_id": mcu,
                    "soh": w.soh,
                    "ambient_temperature": w.amb,
                    "mean_neg_current": w.mnc,
                    "datetime": dt.isoformat(),
                    "discharge_cycle_index": dci,
                    "protocol": w.protocol,
                    "phase": phase,
                    "total_rows": len(new_idx),
                },
            )
            dci += 1
        return dci
