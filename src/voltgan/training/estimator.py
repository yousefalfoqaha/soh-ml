from __future__ import annotations

import json
import signal

import torch
from torch.utils.data import DataLoader

from voltgan.config import (
    BATCH_SIZE,
    ESTIMATOR_CHECKPOINT_PATH,
    LEARNING_RATE,
    MAX_SEQUENCE_LENGTH,
    N_EPOCHS,
    PHASE_ORDER,
    RANDOM_SEED,
    REFERENCE_DISCHARGE_RATE,
    REFERENCE_TEMPERATURE,
    STATS_PATH,
    TESTING_MCUS,
    TRAINING_MCUS,
    VALIDATION_MCUS,
    WUPPERTAL_PROVIDER,
)
from voltgan.dataset import (
    DatasetAnalyzer,
    EstimatorDataset,
    InstanceRepository,
    SohCurveFitter,
)
from voltgan.models import SohEstimatorClient

_interrupted = False


def _handle_sigint(sig, frame):
    global _interrupted
    print("\nInterrupt received, finishing current epoch...")
    _interrupted = True


signal.signal(signal.SIGINT, _handle_sigint)


def _worker_init(worker_id):
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _print_dataset_diagnostics(name: str, instances: list, dataset: EstimatorDataset):
    total_samples = sum(len(inst) for inst in instances)
    avg_len = total_samples / max(len(instances), 1)

    print(f"\n--- {name} Diagnostics ---")
    print(f"Total Instances (Files): {len(instances)}")
    print(f"Total Sequence Length (Sum of rows): {total_samples:,}")
    print(f"Average Length per Instance: {avg_len:,.1f} rows")
    print(f"Total Windows Generated (for DataLoader): {len(dataset):,}")
    print("-" * 25)


def _print_validation_stats(val_instances: list):
    print("\n" + "=" * 55)
    print(" VALIDATION SET DETAILED STATISTICS")
    print("=" * 55)

    # 1. Temperature Distribution
    dist = DatasetAnalyzer.compute_temp_distribution(val_instances, PHASE_ORDER)
    print("\n--- Temperature Band Distribution ---")
    header = f"{'Phase':<15} | " + " | ".join(f"{tc:^5}C" for tc in dist.temp_bands)
    print(header)
    print("-" * len(header))
    for phase, row_counts in zip(dist.phase_order, dist.matrix):
        counts_str = " | ".join(f"{c:^6}" for c in row_counts)
        print(f"{phase:<15} | {counts_str}")
    print("-" * len(header))
    totals_str = " | ".join(f"{t:^6}" for t in dist.col_totals)
    print(f"{'Total':<15} | {totals_str}")

    # 2. MCU Summaries
    fitter = SohCurveFitter(
        reference_temperature=REFERENCE_TEMPERATURE,
        reference_discharge_rate=REFERENCE_DISCHARGE_RATE,
    )
    summaries = DatasetAnalyzer.compute_mcu_summaries(val_instances, fitter)
    print("\n--- MCU SoH Range & Cycles ---")
    print(f"{'MCU':<10} | {'SoH Range':<15} | {'Cycles':<10}")
    print("-" * 45)
    for rec in summaries:
        soh_range = f"{rec.soh_max * 100:.1f}% - {rec.soh_min * 100:.1f}%"
        print(f"{rec.mcu_id:<10} | {soh_range:<15} | {rec.cycles:<10}")
    print("=" * 55 + "\n")


def main() -> None:
    torch.manual_seed(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    wuppertal_repo = InstanceRepository(provider=WUPPERTAL_PROVIDER)

    print("\nLoading Training Data...")
    train_instances = wuppertal_repo.load(TRAINING_MCUS, max_length=MAX_SEQUENCE_LENGTH)
    # Force absolute deterministic ordering
    train_instances.sort(key=lambda inst: inst.filepath.name)

    print("Loading Validation Data...")
    val_mcus = VALIDATION_MCUS + TESTING_MCUS
    val_instances = wuppertal_repo.load(val_mcus, max_length=MAX_SEQUENCE_LENGTH)
    # Force absolute deterministic ordering
    val_instances.sort(key=lambda inst: inst.filepath.name)

    # Print the specific validation statistics you requested
    _print_validation_stats(val_instances)

    with open(STATS_PATH) as f:
        stats = json.load(f)

    training_dataset = EstimatorDataset(instances=train_instances, stats=stats)
    validation_dataset = EstimatorDataset(instances=val_instances, stats=stats)

    _print_dataset_diagnostics("Training", train_instances, training_dataset)
    _print_dataset_diagnostics("Validation", val_instances, validation_dataset)

    training_loader = DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=_worker_init,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=_worker_init,
    )

    client = SohEstimatorClient(device=device, checkpoint_path=None, is_training=True)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(client.model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=15,
        factor=0.5,
        cooldown=2,
        min_lr=1e-6,
    )
    patience = 50

    print(
        f"\nTrain batches: {len(training_loader)} | "
        f"Validation batches: {len(validation_loader)}"
    )
    print(
        f"Starting training for {N_EPOCHS} epochs "
        f"(early stopping patience={patience})..."
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    prev_lr = optimizer.param_groups[0]["lr"]

    global _interrupted

    for epoch in range(N_EPOCHS):
        total_train_loss = 0.0
        total_val_loss = 0.0

        client.train()
        for X, conditions, y in training_loader:
            if _interrupted:
                break

            X = X.to(device, non_blocking=True)
            conditions = conditions.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            y_pred = client(X, conditions)
            loss = criterion(y_pred, y)
            loss.backward()
            total_train_loss += loss.item()
            optimizer.step()

        if _interrupted:
            break

        client.eval()
        with torch.no_grad():
            for X, conditions, y in validation_loader:
                if _interrupted:
                    break

                X = X.to(device, non_blocking=True)
                conditions = conditions.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                y_pred = client(X, conditions)
                loss = criterion(y_pred, y)
                total_val_loss += loss.item()

        avg_train = total_train_loss / len(training_loader)
        avg_val = total_val_loss / len(validation_loader)
        scheduler.step(avg_val)

        cur_lr = optimizer.param_groups[0]["lr"]
        lr_marker = " v" if cur_lr < prev_lr - 1e-12 else ""
        prev_lr = cur_lr

        print(
            f"Epoch {epoch + 1:02d}/{N_EPOCHS} | "
            f"lr={cur_lr:.2e}{lr_marker} | "
            f"Train Loss: {avg_train:.5f} | "
            f"Valid Loss: {avg_val:.5f}"
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.clone() for k, v in client.model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(no improvement for {patience} epochs)."
                )
                break

        if _interrupted:
            break

    if best_state is not None:
        client.model.load_state_dict(best_state)
        print(f"Restored best model (val loss={best_val_loss:.5f}).")

    torch.save(client.model.state_dict(), ESTIMATOR_CHECKPOINT_PATH)
    print(f"Model saved -> {ESTIMATOR_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
