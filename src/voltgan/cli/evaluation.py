from __future__ import annotations

import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from voltgan.config import (
    CONFERENCE_DIR,
    ESTIMATOR_CHECKPOINT_PATH,
    EVALUATION_PROVIDER,
    FINETUNED_CHECKPOINT_PATH,
    MAX_SEQUENCE_LENGTH,
    OXFORD_TESTING_MCUS,
    PHASE_ORDER,
    PROTOCOL_ORDER,
    RANDOM_SEED,
    REFERENCE_DISCHARGE_RATE,
    REFERENCE_TEMPERATURE,
    STATS_PATH,
    TESTING_MCUS,
    TRAINING_PROVIDER,
)
from voltgan.dataset import EstimatorDataset, InstanceRepository, SohCurveFitter
from voltgan.evaluation import InferenceEngine
from voltgan.evaluation.metrics import MetricsAggregator
from voltgan.evaluation.pfi import FeatureSpec, PermutationImportanceEvaluator
from voltgan.models import SohEstimatorClient
from voltgan.utils.latex import HLine, LatexTable, SectionHeader, TableRow

_TABLE_HEADER = [
    "SoH (\\%)",
    "RMSE",
    "MAE",
    "R\\textsuperscript{2}",
    "\\%Err",
    "Cycles",
]

_PFI_FEATURE_SPECS = [
    FeatureSpec("$V$", "X", (0,)),
    FeatureSpec("$I$", "X", (1,)),
    FeatureSpec("$T$", "X", (2,)),
    FeatureSpec("$T_{amb}$", "cond", ()),
]


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    with open(STATS_PATH) as f:
        stats = json.load(f)

    client = SohEstimatorClient(
        device=device, checkpoint_path=ESTIMATOR_CHECKPOINT_PATH
    )

    repo = InstanceRepository(provider=TRAINING_PROVIDER)

    instances = repo.load(TESTING_MCUS, max_length=MAX_SEQUENCE_LENGTH)
    print(f"Loaded {len(instances)} test instances from {TESTING_MCUS}")

    dataset = EstimatorDataset(instances, stats)
    engine = InferenceEngine(client=client, dataset=dataset, stats=stats)
    results = engine.run_predictions()

    overall_metrics = MetricsAggregator.compute("Overall", results)

    # baseline results table
    phase_groups = defaultdict(list)
    for r in results:
        phase_groups[r.instance.phase].append(r)

    phase_rows = [
        TableRow(
            cells=MetricsAggregator.compute(phase, phase_groups[phase]).to_latex_cells()
        )
        for phase in PHASE_ORDER
        if phase in phase_groups
    ]

    LatexTable(
        out_path=CONFERENCE_DIR / "baseline_results.tex",
        caption="BASELINE ESTIMATOR PERFORMANCE",
        label="tab:baseline_results",
        align="lcccccc",
        headers=["Phase", *_TABLE_HEADER],
        items=[
            *phase_rows,
            HLine(),
            TableRow(cells=overall_metrics.to_latex_cells(), bold=True),
        ],
    ).write()

    # temperature & protocol results table
    temp_groups = defaultdict(list)
    proto_groups = defaultdict(list)
    for r in results:
        temp_groups[r.instance.temp_center].append(r)
        proto_groups[r.instance.protocol].append(r)

    temp_rows = [
        TableRow(
            cells=MetricsAggregator.compute(
                rf"${tc}^{{\circ}}\text{{C}}$", temp_groups[tc]
            ).to_latex_cells()
        )
        for tc in sorted(temp_groups)
    ]

    proto_rows = [
        TableRow(
            cells=MetricsAggregator.compute(proto, proto_groups[proto]).to_latex_cells()
        )
        for proto in PROTOCOL_ORDER
        if proto in proto_groups
    ]

    LatexTable(
        out_path=CONFERENCE_DIR / "temp_protocol_results.tex",
        caption="ESTIMATOR PERFORMANCE BY TEMPERATURE AND PROTOCOL",
        label="tab:temp_protocol_results",
        align="lcccccc",
        headers=["Slice", *_TABLE_HEADER],
        items=[
            SectionHeader(title="Temperature Bands"),
            *temp_rows,
            HLine(),
            SectionHeader(title="Discharge Protocols"),
            *proto_rows,
            HLine(),
            TableRow(cells=overall_metrics.to_latex_cells(), bold=True),
        ],
    ).write()

    # oxford: zero-shot vs. fine-tuned, evaluated on the held-out test set only
    oxford_repo = InstanceRepository(provider=EVALUATION_PROVIDER)
    oxford_instances = oxford_repo.load(OXFORD_TESTING_MCUS)
    print(f"Loaded {len(oxford_instances)} Oxford instances from {OXFORD_TESTING_MCUS}")
    print(f"Evaluating on {len(oxford_instances)} held-out Oxford test instances")

    oxford_eval_dataset = EstimatorDataset(oxford_instances, stats)

    # zero-shot: the base Wuppertal-trained model, never seen Oxford data
    zero_shot_engine = InferenceEngine(
        client=client, dataset=oxford_eval_dataset, stats=stats
    )
    zero_shot_results = zero_shot_engine.run_predictions()
    zero_shot_metrics = MetricsAggregator.compute("Zero-Shot", zero_shot_results)

    # fine-tuned: same architecture, weights adapted on the Oxford fine-tune split
    if not FINETUNED_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Fine-tuned checkpoint not found at {FINETUNED_CHECKPOINT_PATH}. "
            "Run the fine-tuning script before evaluating."
        )
    finetuned_client = SohEstimatorClient(
        device=device, checkpoint_path=FINETUNED_CHECKPOINT_PATH
    )
    finetuned_engine = InferenceEngine(
        client=finetuned_client, dataset=oxford_eval_dataset, stats=stats
    )
    finetuned_results = finetuned_engine.run_predictions()
    finetuned_metrics = MetricsAggregator.compute("Fine-Tuned", finetuned_results)

    LatexTable(
        out_path=CONFERENCE_DIR / "oxford_results.tex",
        caption="OXFORD DATASET PERFORMANCE: ZERO-SHOT VS. FINE-TUNED "
        "(EVALUATED ON HELD-OUT TEST CELLS)",
        label="tab:oxford_results",
        align="lcccccc",
        headers=["Model", *_TABLE_HEADER],
        items=[
            TableRow(cells=zero_shot_metrics.to_latex_cells()),
            TableRow(cells=finetuned_metrics.to_latex_cells(), bold=True),
        ],
    ).write()

    # testing trajectory plotting (Updated to TESTING_MCUS)
    test_results = [r for r in results if r.instance.cell_id in TESTING_MCUS]
    if test_results:
        fitter = SohCurveFitter(
            reference_temperature=REFERENCE_TEMPERATURE,
            reference_discharge_rate=REFERENCE_DISCHARGE_RATE,
        )

        test_instances = repo.load(TESTING_MCUS)
        fit_result = fitter.fit(test_instances)

        if fit_result is not None:
            print(
                f"Loaded {len(fit_result.ref_points)} reference points from {TESTING_MCUS}"
            )

            fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")

            ref_dci = np.array([p[0] for p in fit_result.ref_points], dtype=float)
            pred_dci = np.array([r.instance.dci for r in test_results], dtype=float)

            # Safely extract predictions depending on InferenceEngine result structure
            pred_soh = np.array(
                [
                    getattr(
                        r,
                        "prediction",
                        getattr(r, "predicted_soh", getattr(r, "pred", 0.0)),
                    )
                    for r in test_results
                ],
                dtype=float,
            )

            min_dci = float(ref_dci.min())
            max_dci = float(max(ref_dci.max(), pred_dci.max()))
            dense_dci = np.linspace(min_dci, max_dci, 500)
            dense_soh = np.clip(
                [fit_result.model(float(t)) for t in dense_dci], 0.0, 1.0
            )

            ax.plot(
                dense_dci,
                dense_soh,
                color="tab:blue",
                lw=2,
                label="Real (Fitted) SoH",
                zorder=2,
            )

            ax.scatter(
                pred_dci,
                pred_soh,
                s=30,
                color="tab:red",
                alpha=0.7,
                label="Predicted SoH",
                zorder=3,
                edgecolors="none",
            )

            ax.set_xlabel("Discharge Cycles")
            ax.set_ylabel("SoH")
            ax.legend(fontsize=9, loc="best")
            ax.grid(True, alpha=0.3)

            out_traj = CONFERENCE_DIR / "test_trajectory.pdf"
            fig.savefig(out_traj, bbox_inches="tight")
            plt.close(fig)
            print(f"Plot saved -> {out_traj}")
        else:
            print("Not enough reference points to fit SoH curve for trajectory plot.")

    # stratified permutation feature importance (pfi)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    pfi_eval = PermutationImportanceEvaluator(engine=engine, dataset=dataset, repeats=5)
    report = pfi_eval.run(features=_PFI_FEATURE_SPECS, protocols=PROTOCOL_ORDER)

    LatexTable(
        out_path=CONFERENCE_DIR / "pfi_results.tex",
        caption="STRATIFIED PFI RESULTS",
        label="tab:pfi_results",
        align="l" + "r" * len(report.ranked_features),
        headers=["Feature", *[f.name for f in report.ranked_features]],
        items=[*report.to_latex_delta_rows(), HLine()],
    ).write()

    # build pfi baseline table
    pfi_baseline_items = []
    for p in PROTOCOL_ORDER:
        if p in report.baseline_metrics and report.baseline_metrics[p].cycles > 0:
            c = report.baseline_metrics[p].to_latex_cells()
            # MetricSet cell order: [label, soh, rmse, mae, r2, pct_err, cycles]
            pfi_baseline_items.append(TableRow(cells=[c[0], c[6], c[2], c[3], c[5]]))

    c_overall = overall_metrics.to_latex_cells()
    pfi_baseline_items.extend(
        [
            HLine(),
            TableRow(
                cells=[
                    c_overall[0],
                    c_overall[6],
                    c_overall[2],
                    c_overall[3],
                    c_overall[5],
                ],
                bold=True,
            ),
        ]
    )

    LatexTable(
        out_path=CONFERENCE_DIR / "pfi_baseline.tex",
        caption="PER-PROTOCOL BASELINE ESTIMATOR PERFORMANCE",
        label="tab:pfi_baseline",
        align="lcccc",
        headers=["Protocol", "Cycles", "RMSE", "MAE", r"\%Err"],
        items=pfi_baseline_items,
    ).write()

    # pfi chart plotting
    active_protos = report.active_protocols
    ranked_features = report.ranked_features

    feature_labels = [f.name for f in ranked_features]
    group_centers = np.arange(len(feature_labels))
    bar_width = 0.8 / max(len(active_protos), 1)

    cmap = plt.get_cmap("tab10")
    proto_colors = {p: cmap(i) for i, p in enumerate(active_protos)}

    fig, ax = plt.subplots(figsize=(8.5, 4.5), layout="constrained")
    for j, p in enumerate(active_protos):
        bar_deltas, bar_errs = [], []
        for feature in ranked_features:
            res = report.results[feature.name].get(p)
            if res and res.deltas:
                bar_deltas.append(float(np.mean(res.deltas)))
                bar_errs.append(
                    float(np.std(res.deltas, ddof=1)) if len(res.deltas) > 1 else 0.0
                )
            else:
                bar_deltas.append(0.0)
                bar_errs.append(0.0)

        offsets = (j - (len(active_protos) - 1) / 2) * bar_width
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

    CONFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    out = CONFERENCE_DIR / "pfi_importance.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved -> {out}")


if __name__ == "__main__":
    main()
