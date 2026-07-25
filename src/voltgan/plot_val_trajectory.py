from __future__ import annotations

import argparse
import json
from collections import defaultdict

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from voltgan.config import (
    BATCH_SIZE,
    CONFERENCE_PATH,
    ESTIMATOR_BASE_CHANNELS,
    ESTIMATOR_CHECKPOINT_PATH,
    ESTIMATOR_GRU_HIDDEN_SIZE,
    ESTIMATOR_GRU_N_LAYERS,
    ESTIMATOR_INPUT_FEATURES,
    ESTIMATOR_KERNEL_SIZE,
    ESTIMATOR_N_CONDITIONS,
    ESTIMATOR_STRIDE,
    HDF_ROOT,
    STATS_PATH,
    VALIDATION_MCUS,
)
from voltgan.data import EstimatorDataset
from voltgan.models import SohEstimator
from voltgan.pipeline.soh_curve import fit_soh_curve
from voltgan.utils.discover import load_instances
from voltgan.utils.reference import load_reference_points


def _run_inference(mcus: list[str], stats: dict, device: str):
    model = SohEstimator(
        input_features=ESTIMATOR_INPUT_FEATURES,
        n_conditions=ESTIMATOR_N_CONDITIONS,
        base_channels=ESTIMATOR_BASE_CHANNELS,
        stride=ESTIMATOR_STRIDE,
        kernel_size=ESTIMATOR_KERNEL_SIZE,
        gru_hidden_size=ESTIMATOR_GRU_HIDDEN_SIZE,
        gru_n_layers=ESTIMATOR_GRU_N_LAYERS,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(torch.load(ESTIMATOR_CHECKPOINT_PATH, map_location=device))
    model.eval()

    instances = load_instances(HDF_ROOT, mcus)
    dataset = EstimatorDataset(instances, stats)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_preds = []
    with torch.no_grad():
        for batch_X, batch_cond, _ in loader:
            preds = model(batch_X.to(device), batch_cond.to(device)).squeeze(-1)
            all_preds.append(preds.cpu().numpy())
    all_preds = np.concatenate(all_preds)

    soh_mean = stats["soh"]["mean"]
    soh_std = stats["soh"]["standard_deviation"]
    preds_destd = all_preds * soh_std + soh_mean

    instance_preds: dict[int, list[float]] = defaultdict(list)
    for i, (inst, _, _) in enumerate(dataset.windows):
        instance_preds[id(inst)].append(float(preds_destd[i]))

    per_instance = []
    for inst in instances:
        preds = instance_preds.get(id(inst))
        if not preds:
            continue
        mean_pred = float(np.mean(preds))
        with h5py.File(inst.filepath, "r") as f:
            dci = float(f.attrs.get("discharge_cycle_index", 0))
        per_instance.append((dci, mean_pred))

    return per_instance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot validation MCU SoH trajectory with predicted points."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="'cuda' or 'cpu'. Auto-detected if omitted.",
    )
    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ref_points = load_reference_points(VALIDATION_MCUS, HDF_ROOT)
    print(f"Loaded {len(ref_points)} reference points from {VALIDATION_MCUS}")

    fit = fit_soh_curve(ref_points)
    if fit is None:
        raise ValueError("Not enough reference points to fit SoH curve.")
    model, first_dci, last_dci, first_soh, last_soh = fit

    with open(STATS_PATH) as f:
        stats = json.load(f)

    per_instance = _run_inference(VALIDATION_MCUS, stats, device)
    print(f"Ran inference on {len(per_instance)} instances")

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")

    ref_dci = np.array([p[0] for p in ref_points], dtype=float)
    pred_dci = np.array([p[0] for p in per_instance], dtype=float)
    pred_soh = np.array([p[1] for p in per_instance], dtype=float)
    min_dci = float(ref_dci.min())
    max_dci = float(max(ref_dci.max(), pred_dci.max()))
    dense_dci = np.linspace(min_dci, max_dci, 500)
    dense_soh = np.clip([model(float(t)) for t in dense_dci], 0.0, 1.0)
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

    CONFERENCE_PATH.mkdir(parents=True, exist_ok=True)
    out = CONFERENCE_PATH / "val_trajectory.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved -> {out}")


if __name__ == "__main__":
    main()
