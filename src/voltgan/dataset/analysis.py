from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from voltgan.dataset.instance import DischargeInstance
from voltgan.dataset.soh_curve import SohCurveFitter


@dataclass
class TempDistributionMatrix:
    phase_order: list[str]
    temp_bands: list[int]
    matrix: list[list[int]]
    col_totals: list[int]


@dataclass
class McuSummaryRecord:
    mcu_id: str
    soh_min: float
    soh_max: float
    cycles: int

    def to_latex_cells(self) -> list[str]:
        mcu_label = self.mcu_id.replace("mcu", "").replace("cell", "")
        soh_range = f"${self.soh_max * 100:.1f}$--${self.soh_min * 100:.1f}$"
        return [mcu_label, soh_range, str(self.cycles)]


class DatasetAnalyzer:
    @staticmethod
    def compute_temp_distribution(
        instances: list[DischargeInstance], phase_order: list[str]
    ) -> TempDistributionMatrix:
        counts = defaultdict(int)
        for inst in instances:
            counts[(inst.phase, inst.temp_center)] += 1

        temp_bands = sorted({tc for (_, tc) in counts})
        matrix = [
            [counts.get((phase, tc), 0) for tc in temp_bands] for phase in phase_order
        ]
        col_totals = [
            sum(matrix[i][j] for i in range(len(phase_order)))
            for j in range(len(temp_bands))
        ]

        return TempDistributionMatrix(
            phase_order=phase_order,
            temp_bands=temp_bands,
            matrix=matrix,
            col_totals=col_totals,
        )

    @staticmethod
    def compute_mcu_summaries(
        instances: list[DischargeInstance], fitter: SohCurveFitter
    ) -> list[McuSummaryRecord]:
        by_mcu = defaultdict(list)
        for instance in instances:
            by_mcu[instance.cell_id].append(instance)

        def _get_num(mcu_str: str) -> int:
            match = re.search(r"\d+", mcu_str)
            return int(match.group()) if match else 0

        records = []
        for mcu_name, insts in sorted(by_mcu.items(), key=lambda x: _get_num(x[0])):
            if not insts:
                continue

            ref_instances = fitter.filter_reference(insts)

            if not ref_instances:
                continue

            soh_values = [inst.soh for inst in ref_instances]

            records.append(
                McuSummaryRecord(
                    mcu_id=mcu_name,
                    soh_min=min(soh_values),
                    soh_max=max(soh_values),
                    cycles=len(insts),
                )
            )

        return records
