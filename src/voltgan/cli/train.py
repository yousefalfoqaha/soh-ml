from __future__ import annotations

import json
import signal
import sys

import torch
from torch.utils.data import DataLoader

from voltgan.config import (
    BATCH_SIZE,
    DROPOUT,
    ESTIMATOR_BASE_CHANNELS,
    ESTIMATOR_CHECKPOINT_PATH,
    ESTIMATOR_GRU_HIDDEN_SIZE,
    ESTIMATOR_GRU_N_LAYERS,
    ESTIMATOR_INPUT_FEATURES,
    ESTIMATOR_KERNEL_SIZE,
    ESTIMATOR_N_CONDITIONS,
    ESTIMATOR_STRIDE,
    LEARNING_RATE,
    MAX_SEQUENCE_LENGTH,
    N_EPOCHS,
    RANDOM_SEED,
    STATS_PATH,
    TESTING_MCUS,
    TRAINING_MCUS,
    VALIDATION_MCUS,
    WUPPERTAL_PROVIDER,
)
from voltgan.dataset import EstimatorDataset, InstanceRepository
from voltgan.models import SohEstimator

_interrupted = False
_force_quit = False


def _handle_sigint(sig, frame):
    global _interrupted, _force_quit

    if _interrupted:
        print(
            "\nSecond interrupt received! Exiting immediately without saving current epoch..."
        )
        _force_quit = True
        sys.exit(1)

    _interrupted = True
    print(
        "\n[!] Graceful Stop Requested: Finishing the current epoch... (Press Ctrl+C again to force quit)"
    )


def main() -> None:
    signal.signal(signal.SIGINT, _handle_sigint)

    torch.manual_seed(RANDOM_SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    repo = InstanceRepository(provider=WUPPERTAL_PROVIDER)

    train_instances = repo.load(TRAINING_MCUS, max_length=MAX_SEQUENCE_LENGTH)
    val_mcus = VALIDATION_MCUS + TESTING_MCUS
    val_instances = repo.load(val_mcus, max_length=MAX_SEQUENCE_LENGTH)

    with open(STATS_PATH) as f:
        stats = json.load(f)

    training_dataset = EstimatorDataset(instances=train_instances, stats=stats)
    validation_dataset = EstimatorDataset(instances=val_instances, stats=stats)

    training_loader = DataLoader(
        training_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True
    )

    model = SohEstimator(
        input_features=ESTIMATOR_INPUT_FEATURES,
        n_conditions=ESTIMATOR_N_CONDITIONS,
        base_channels=ESTIMATOR_BASE_CHANNELS,
        stride=ESTIMATOR_STRIDE,
        gru_hidden_size=ESTIMATOR_GRU_HIDDEN_SIZE,
        gru_n_layers=ESTIMATOR_GRU_N_LAYERS,
        dropout=DROPOUT,
        kernel_size=ESTIMATOR_KERNEL_SIZE,
    ).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=15,
        factor=0.5,
        cooldown=2,
        min_lr=1e-5,
    )

    patience = 50
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    prev_lr = optimizer.param_groups[0]["lr"]

    try:
        for epoch in range(N_EPOCHS):
            model.train()
            total_train_loss = 0.0
            total_train_samples = 0

            for X, conditions, y in training_loader:
                X = X.to(device, non_blocking=True)
                conditions = conditions.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                y_pred = model(X, conditions)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * y.size(0)
                total_train_samples += y.size(0)

            avg_train = total_train_loss / total_train_samples

            model.eval()
            total_val_loss = 0.0
            total_val_samples = 0

            with torch.no_grad():
                for X, conditions, y in validation_loader:
                    X = X.to(device, non_blocking=True)
                    conditions = conditions.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)

                    y_pred = model(X, conditions)
                    loss = criterion(y_pred, y)

                    total_val_loss += loss.item() * y.size(0)
                    total_val_samples += y.size(0)

            avg_val = total_val_loss / total_val_samples

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
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1}...")
                    break

            if _interrupted:
                break

    finally:
        if not _force_quit:
            if best_state is not None:
                model.load_state_dict(best_state)
                print(f"Restored best model (val loss={best_val_loss:.5f}).")

            torch.save(model.state_dict(), ESTIMATOR_CHECKPOINT_PATH)
            print(f"Model saved -> {ESTIMATOR_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
