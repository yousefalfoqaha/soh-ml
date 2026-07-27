from __future__ import annotations

import json
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from voltgan.config import (
    CONFERENCE_PATH,
    ESTIMATOR_CHECKPOINT_PATH,
    HDF_ROOT,
    OXFORD_MCUS,
    PHASE_ORDER,
    PROTOCOL_ORDER,
    RANDOM_SEED,
    STATS_PATH,
    TESTING_MCUS,
    VALIDATION_MCUS,
)
from voltgan.dataset import EstimatorDataset, InstanceRepository
from voltgan.dataset.instance import DischargeInstance
from voltgan.evaluation import (
    InferenceEngine,
    MetricsAggregator,
    MetricSet,
    PredictionResult,
    aggregate_per_instance,
)
from voltgan.models import SohEstimatorClient
from voltgan.utils import LatexTableWriter

_OXFORD_FILE_PATTERN = re.compile(r"oxford_cell(\d+)_cyc\d{4}\.hdf$")

_PFI_FEATURE_SPECS: list[tuple[str, str, tuple[int, ...]]] = [
    ("$V$", "X", (0,)),
    ("$I$", "X", (1,)),
    ("$T$", "X", (2,)),
    ("$T_{amb}$", "cond", ()),
]
_PFI_REPEATS = 5

_TABLE_HEADER = [
    "SoH (\\%)",
    "RMSE",
    "MAE",
    "R\\textsuperscript{2}",
    "\\%Err",
    "Cycles",
]


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    with open(STATS_PATH) as f:
        stats = json.load(f)

    client = SohEstimatorClient(
        device=device, checkpoint_path=ESTIMATOR_CHECKPOINT_PATH
    )
    repo = InstanceRepository(root=HDF_ROOT)

    instances = repo.load(VALIDATION_MCUS + TESTING_MCUS)
    print(
        f"Loaded {len(instances)} valid/test instances from {VALIDATION_MCUS + TESTING_MCUS}"
    )

    dataset = EstimatorDataset(instances, stats)
    engine = InferenceEngine(client=client, dataset=dataset, stats=stats)
    results = engine.run_predictions()

    _write_baseline_results(results)
    _write_temp_protocol_results(results)

    baseline, overall, deltas, active = _run_pfi(engine, dataset)
    _write_pfi_tables(baseline, overall, deltas, active)
    _plot_pfi_chart(deltas, active)

    instances = repo.load(OXFORD_MCUS)
    print(f"Loaded {len(instances)} Oxford instances")
    dataset = EstimatorDataset(instances, stats)
    engine = InferenceEngine(client=client, dataset=dataset, stats=stats)
    _write_oxford_results(engine.run_predictions())


def _write_baseline_results(results: list[PredictionResult]) -> None:
    overall = MetricsAggregator.compute("Overall", results)

    phase_groups: dict[str, list[PredictionResult]] = defaultdict(list)
    for r in results:
        phase_groups[r.instance.phase].append(r)

    rows = [
        MetricsAggregator.compute(phase, phase_groups[phase]).cells()
        for phase in PHASE_ORDER
        if phase_groups.get(phase)
    ]

    (
        LatexTableWriter(CONFERENCE_PATH / "baseline_results.tex")
        .caption("BASELINE ESTIMATOR PERFORMANCE")
        .label("tab:baseline_results")
        .align("lcccccc")
        .hline()
        .header(["Phase", *_TABLE_HEADER])
        .rows(rows)
        .hline()
        .bold_row(overall.cells(bold=True))
        .write()
    )


def _write_temp_protocol_results(results: list[PredictionResult]) -> None:
    overall = MetricsAggregator.compute("Overall", results)

    temp_groups: dict[int, list[PredictionResult]] = defaultdict(list)
    proto_groups: dict[str, list[PredictionResult]] = defaultdict(list)
    for r in results:
        temp_groups[r.instance.temp_center].append(r)
        proto_groups[r.instance.protocol].append(r)

    w = (
        LatexTableWriter(CONFERENCE_PATH / "temp_protocol_results.tex")
        .caption("ESTIMATOR PERFORMANCE BY TEMPERATURE AND PROTOCOL")
        .label("tab:temp_protocol_results")
        .align("lcccccc")
        .hline()
        .header(["Slice", *_TABLE_HEADER])
        .hline()
    )

    w.section("Temperature Bands")
    for tc in sorted(temp_groups):
        label = rf"${tc}^{{\circ}}\text{{C}}$"
        w.row(MetricsAggregator.compute(label, temp_groups[tc]).cells())

    w.hline()

    w.section("Discharge Protocols")
    for proto in PROTOCOL_ORDER:
        group = proto_groups.get(proto)
        if not group:
            continue
        w.row(MetricsAggregator.compute(proto, group).cells())

    w.hline()

    w.bold_row(overall.cells(bold=True))

    w.write()


def _run_pfi(
    engine: InferenceEngine, dataset: EstimatorDataset
) -> tuple[
    dict[str, MetricSet],
    MetricSet,
    dict[str, dict[str, list[float]]],
    list[str],
]:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    X, conditions, window_to_inst, window_to_proto = dataset.materialize()

    inst_by_id: dict[int, DischargeInstance] = {}
    for inst, _, _ in dataset.windows:
        inst_by_id[id(inst)] = inst

    base_preds = engine.predict_tensors(X, conditions)
    base_inst_pred = aggregate_per_instance(base_preds, window_to_inst)

    all_results = [
        PredictionResult(inst_by_id[iid], pred) for iid, pred in base_inst_pred.items()
    ]
    overall = MetricsAggregator.compute("Overall", all_results)

    protocol_windows: dict[str, list[int]] = defaultdict(list)
    for w_idx, proto in enumerate(window_to_proto):
        protocol_windows[proto].append(w_idx)
    protocol_instance_ids: dict[str, set[int]] = {
        p: {window_to_inst[i] for i in idxs} for p, idxs in protocol_windows.items()
    }

    baseline: dict[str, MetricSet] = {}
    for p in PROTOCOL_ORDER:
        ids = protocol_instance_ids.get(p, set())
        results = [
            PredictionResult(inst_by_id[iid], base_inst_pred[iid])
            for iid in ids
            if iid in base_inst_pred
        ]
        baseline[p] = MetricsAggregator.compute(p, results)

    print(f"Ran baseline inference ({overall.cycles} instances)")

    deltas: dict[str, dict[str, list[float]]] = {
        f: {p: [] for p in PROTOCOL_ORDER} for f, _, _ in _PFI_FEATURE_SPECS
    }

    for fname, tensor_type, channels in _PFI_FEATURE_SPECS:
        for p in PROTOCOL_ORDER:
            bucket_idx = protocol_windows.get(p, [])
            if not bucket_idx or baseline[p].cycles == 0:
                continue
            for _ in range(_PFI_REPEATS):
                X_perm = X.clone()
                cond_perm = conditions.clone()
                local = torch.randperm(len(bucket_idx))
                if tensor_type == "X":
                    for ch in channels:
                        src = X_perm[bucket_idx, :, ch]
                        X_perm[bucket_idx, :, ch] = src[local]
                else:
                    src = cond_perm[bucket_idx, 0]
                    cond_perm[bucket_idx, 0] = src[local]

                perm_preds = engine.predict_tensors(X_perm, cond_perm)
                perm_inst_pred = aggregate_per_instance(perm_preds, window_to_inst)
                perm_results = [
                    PredictionResult(inst_by_id[iid], perm_inst_pred[iid])
                    for iid in protocol_instance_ids[p]
                    if iid in perm_inst_pred
                ]
                perm_metric = MetricsAggregator.compute(p, perm_results)
                if perm_metric.cycles > 0:
                    deltas[fname][p].append(perm_metric.rmse - baseline[p].rmse)

    active = [p for p in PROTOCOL_ORDER if protocol_windows.get(p)]
    return baseline, overall, deltas, active


def _write_pfi_tables(
    baseline: dict[str, MetricSet],
    overall: MetricSet,
    deltas: dict[str, dict[str, list[float]]],
    active: list[str],
) -> None:
    ranked = sorted(
        _PFI_FEATURE_SPECS,
        key=lambda spec: _mean_delta_rmse(spec[0], active, deltas),
        reverse=True,
    )

    align = "l" + "r" * len(active)
    w = (
        LatexTableWriter(CONFERENCE_PATH / "pfi_results.tex")
        .caption("STRATIFIED PFI RESULTS")
        .label("tab:pfi_results")
        .align(align)
        .hline()
        .header(["Feature", *active])
        .hline()
    )
    for fname, _, _ in ranked:
        w.row(_pfi_delta_row(fname, active, deltas))
    w.hline()
    w.write()

    base_rows = [
        _pfi_baseline_row(baseline[p]) for p in PROTOCOL_ORDER if baseline[p].cycles > 0
    ]
    base_rows.append(_pfi_baseline_row(overall, bold=True))

    (
        LatexTableWriter(CONFERENCE_PATH / "pfi_baseline.tex")
        .caption("PER-PROTOCOL BASELINE ESTIMATOR PERFORMANCE")
        .label("tab:pfi_baseline")
        .align("lcccc")
        .hline()
        .header(["Protocol", "Cycles", "RMSE", "MAE", "\\%Err"])
        .rows(base_rows)
        .write()
    )


def _plot_pfi_chart(
    deltas: dict[str, dict[str, list[float]]], active: list[str]
) -> None:
    ranked = sorted(
        _PFI_FEATURE_SPECS,
        key=lambda spec: _mean_delta_rmse(spec[0], active, deltas),
        reverse=True,
    )
    feature_labels = [spec[0] for spec in ranked]

    n_features = len(feature_labels)
    n_protocols = len(active)
    group_centers = np.arange(n_features)
    bar_width = 0.8 / max(n_protocols, 1)
    cmap = plt.get_cmap("tab10")
    proto_colors = {p: cmap(i) for i, p in enumerate(active)}

    fig, ax = plt.subplots(figsize=(8.5, 4.5), layout="constrained")
    for j, p in enumerate(active):
        bar_deltas, bar_errs = [], []
        for fname, _, _ in ranked:
            vals = deltas.get(fname, {}).get(p, [])
            if vals:
                bar_deltas.append(float(np.mean(vals)))
                bar_errs.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
            else:
                bar_deltas.append(0.0)
                bar_errs.append(0.0)
        offsets = (j - (n_protocols - 1) / 2) * bar_width
        ax.barh(
            group_centers + offsets,
            bar_deltas,
            height=bar_width,
            xerr=bar_errs,
            color=proto_colors[p],
            edgecolor="black",
            capsize=2,
            label=p,
        )

    ax.set_yticks(group_centers)
    ax.set_yticklabels(feature_labels)
    ax.set_xlabel(r"$\Delta$RMSE (Permuted within Protocol $-$ Baseline)")
    ax.set_title("Stratified PFI")
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(fontsize=9, loc="best", title="Protocol")

    CONFERENCE_PATH.mkdir(parents=True, exist_ok=True)
    out = CONFERENCE_PATH / "pfi_importance.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved -> {out}")


def _write_oxford_results(results: list[PredictionResult]) -> None:
    cell_groups: dict[int, list[PredictionResult]] = defaultdict(list)
    for r in results:
        m = _OXFORD_FILE_PATTERN.search(r.instance.filepath.name)
        if m is None:
            continue
        cell_groups[int(m.group(1))].append(r)

    overall = MetricsAggregator.compute("Overall", results)
    rows = [
        MetricsAggregator.compute(f"Cell {cell}", cell_groups[cell]).cells()
        for cell in sorted(cell_groups)
    ]

    (
        LatexTableWriter(CONFERENCE_PATH / "oxford_results.tex")
        .caption("OXFORD DATASET ZERO-SHOT ESTIMATOR PERFORMANCE")
        .label("tab:oxford_results")
        .align("lcccccc")
        .hline()
        .header(["Cell", *_TABLE_HEADER])
        .rows(rows)
        .hline()
        .bold_row(overall.cells(bold=True))
        .write()
    )


def _mean_delta_rmse(
    feature: str, active: list[str], deltas: dict[str, dict[str, list[float]]]
) -> float:
    vals = [v for p in active for v in deltas.get(feature, {}).get(p, [])]
    return float(np.mean(vals)) if vals else 0.0


def _pfi_baseline_row(metric: MetricSet, *, bold: bool = False) -> list[str]:
    cyc = str(metric.cycles)
    rmse = f"{metric.rmse:.4f}" if metric.cycles > 0 else "--"
    mae = f"{metric.mae:.4f}" if metric.cycles > 0 else "--"
    pct = f"{metric.pct_err:.1f}" + r"\%" if metric.cycles > 0 else "--"
    if bold:
        return [
            rf"\textbf{{{metric.label}}}",
            rf"\textbf{{{cyc}}}",
            rf"$\mathbf{{{metric.rmse:.4f}}}$",
            rf"$\mathbf{{{metric.mae:.4f}}}$",
            rf"$\mathbf{{{metric.pct_err:.1f}\%}}$",
        ]
    return [metric.label, cyc, rmse, mae, pct]


def _pfi_delta_row(
    feature: str, active: list[str], deltas: dict[str, dict[str, list[float]]]
) -> list[str]:
    best_proto: str | None = None
    best_mean = -float("inf")
    for p in active:
        vals = deltas.get(feature, {}).get(p, [])
        if vals:
            m = float(np.mean(vals))
            if m > best_mean:
                best_mean = m
                best_proto = p

    cells = [feature]
    for p in active:
        vals = deltas.get(feature, {}).get(p, [])
        if not vals:
            cells.append("--")
        else:
            m = float(np.mean(vals))
            s = f"+{m:.4f}" if m >= 0 else f"{m:.4f}"
            if p == best_proto and m > 0:
                s = rf"\textbf{{{s}}}"
            cells.append(s)
    return cells
