from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from voltgan.dataset.instance import DischargeInstance


@dataclass(frozen=True)
class Split:
    fine_tune: list[DischargeInstance]
    validation: list[DischargeInstance]
    eval: list[DischargeInstance]


class DatasetSplitter:
    """
    Splits instances per cell along the dci axis into
    fine_tune / validation / eval ranges, driven by percentages.
    """

    def __init__(self, instances: list[DischargeInstance]):
        self.instances = instances

    def split(self, training_percentage: float, validation_percentage: float) -> Split:
        if training_percentage < 0 or validation_percentage < 0:
            raise ValueError("Split percentages must be non-negative.")
        if training_percentage + validation_percentage > 1.0:
            raise ValueError(
                "training_percentage + validation_percentage must not exceed 1.0 "
                f"(got {training_percentage} + {validation_percentage} = "
                f"{training_percentage + validation_percentage})."
            )

        by_cell: dict[str, list[DischargeInstance]] = defaultdict(list)
        for inst in self.instances:
            by_cell[inst.cell_id].append(inst)

        fine_tune: list[DischargeInstance] = []
        validation: list[DischargeInstance] = []
        eval_: list[DischargeInstance] = []

        for cell_id in sorted(by_cell):
            cell_instances = sorted(by_cell[cell_id], key=lambda i: i.dci)
            n = len(cell_instances)

            n_ft = max(1, round(n * training_percentage)) if n > 0 else 0
            n_val = round(n * validation_percentage) if n > 0 else 0

            n_ft = min(n_ft, n)
            n_val = min(n_val, n - n_ft)

            ft_slice = cell_instances[:n_ft]
            val_slice = cell_instances[n_ft : n_ft + n_val]
            eval_slice = cell_instances[n_ft + n_val :]

            fine_tune.extend(ft_slice)
            validation.extend(val_slice)
            eval_.extend(eval_slice)

            print(
                f"  {cell_id}: {n} cycles -> "
                f"{len(ft_slice)} fine_tune / {len(val_slice)} validation / "
                f"{len(eval_slice)} eval"
            )

        return Split(fine_tune=fine_tune, validation=validation, eval=eval_)
