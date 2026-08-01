from __future__ import annotations

import json
import signal
import sys

import torch
from torch.utils.data import DataLoader

from voltgan.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    ESTIMATOR_CHECKPOINT_PATH,
    EVALUATION_PROVIDER,
    FINETUNED_CHECKPOINT_PATH,
    FT_MAX_LEARNING_RATE,
    FT_MIN_LEARNING_RATE,
    MAX_SEQUENCE_LENGTH,
    OXFORD_TRAINING_MCUS,
    OXFORD_VALIDATION_MCUS,
    RANDOM_SEED,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    STATS_PATH,
    WEIGHT_DECAY,
)
from voltgan.dataset import EstimatorDataset, InstanceRepository
from voltgan.models import SohEstimatorClient

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

    repo = InstanceRepository(provider=EVALUATION_PROVIDER)
    train_instances = repo.load(OXFORD_TRAINING_MCUS, max_length=MAX_SEQUENCE_LENGTH)
    val_instances = repo.load(OXFORD_VALIDATION_MCUS, max_length=MAX_SEQUENCE_LENGTH)
    print(
        f"Loaded {len(train_instances)} Oxford instances from {OXFORD_TRAINING_MCUS + OXFORD_VALIDATION_MCUS}"
    )

    with open(STATS_PATH) as f:
        stats = json.load(f)

    train_dataset = EstimatorDataset(instances=train_instances, stats=stats)
    val_dataset = EstimatorDataset(instances=val_instances, stats=stats)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True
    )

    if not ESTIMATOR_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found at {ESTIMATOR_CHECKPOINT_PATH}. "
            "Train the base estimator before fine-tuning."
        )

    client = SohEstimatorClient(
        device=device,
        checkpoint_path=ESTIMATOR_CHECKPOINT_PATH,
        is_training=True,
    )
    client.finetune()
    print(
        f"Loaded pretrained weights and frozen layers from {ESTIMATOR_CHECKPOINT_PATH}"
    )

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(
        client.trainable_parameters(),
        lr=FT_MAX_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=SCHEDULER_PATIENCE,
        factor=SCHEDULER_FACTOR,
        min_lr=FT_MIN_LEARNING_RATE,
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    prev_lr = optimizer.param_groups[0]["lr"]

    try:
        for epoch in range(EPOCHS):
            client.train()
            total_train_loss = 0.0
            total_train_samples = 0

            for X, conditions, y in train_loader:
                X = X.to(device, non_blocking=True)
                conditions = conditions.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                y_pred = client(X, conditions)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * y.size(0)
                total_train_samples += y.size(0)

            avg_train = total_train_loss / total_train_samples

            client.eval()
            total_val_loss = 0.0
            total_val_samples = 0

            with torch.no_grad():
                for X, conditions, y in val_loader:
                    X = X.to(device, non_blocking=True)
                    conditions = conditions.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)

                    y_pred = client(X, conditions)
                    loss = criterion(y_pred, y)

                    total_val_loss += loss.item() * y.size(0)
                    total_val_samples += y.size(0)

            avg_val = total_val_loss / total_val_samples

            scheduler.step(avg_val)
            cur_lr = optimizer.param_groups[0]["lr"]
            lr_marker = " v" if cur_lr < prev_lr - 1e-12 else ""
            prev_lr = cur_lr

            is_best = avg_val < best_val_loss
            star_marker = "*" if is_best else ""

            print(
                f"Epoch {epoch + 1:02d}/{EPOCHS} | "
                f"lr={cur_lr:.2e}{lr_marker} | "
                f"Train Loss: {avg_train:.5f} | "
                f"Valid Loss: {avg_val:.5f}{star_marker}"
            )

            if is_best:
                best_val_loss = avg_val
                best_state = {
                    k: v.clone() for k, v in client.model.state_dict().items()
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping at epoch {epoch + 1}...")
                    break

            if _interrupted:
                break

    finally:
        if not _force_quit:
            if best_state is not None:
                client.model.load_state_dict(best_state)
                print(f"Restored best model (val loss={best_val_loss:.5f}).")

            FINETUNED_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(client.model.state_dict(), FINETUNED_CHECKPOINT_PATH)
            print(f"Model saved -> {FINETUNED_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
