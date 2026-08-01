from __future__ import annotations

import json
import signal
import sys

import torch
from torch.utils.data import DataLoader

from voltgan.config import (
    BATCH_SIZE,
    DROPOUT,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    ESTIMATOR_BASE_CHANNELS,
    ESTIMATOR_CHECKPOINT_PATH,
    ESTIMATOR_GRU_HIDDEN_SIZE,
    ESTIMATOR_GRU_N_LAYERS,
    ESTIMATOR_INPUT_FEATURES,
    ESTIMATOR_KERNEL_SIZE,
    ESTIMATOR_N_CONDITIONS,
    ESTIMATOR_STRIDE,
    EVALUATION_PROVIDER,
    FINETUNED_CHECKPOINT_PATH,
    FT_MAX_LEARNING_RATE,
    FT_MIN_LEARNING_RATE,
    MAX_SEQUENCE_LENGTH,
    OXFORD_MCUS,
    OXFORD_TRAINING_PERCENTAGE,
    OXFORD_VALIDATION_PERCENTAGE,
    RANDOM_SEED,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    STATS_PATH,
    WEIGHT_DECAY,
)
from voltgan.dataset import EstimatorDataset, InstanceRepository
from voltgan.dataset.splitter import OxfordSplitter
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

    repo = InstanceRepository(provider=EVALUATION_PROVIDER)
    oxford_instances = repo.load(OXFORD_MCUS, max_length=MAX_SEQUENCE_LENGTH)
    print(f"Loaded {len(oxford_instances)} Oxford instances from {OXFORD_MCUS}")

    print(
        f"Splitting per cell (chronological, by dci): "
        f"fine_tune={OXFORD_TRAINING_PERCENTAGE:.0%}, "
        f"validation={OXFORD_VALIDATION_PERCENTAGE:.0%}, "
        f"eval={1 - OXFORD_TRAINING_PERCENTAGE - OXFORD_VALIDATION_PERCENTAGE:.0%}"
    )
    splitter = OxfordSplitter(oxford_instances)
    split = splitter.split(
        training_percentage=OXFORD_TRAINING_PERCENTAGE,
        validation_percentage=OXFORD_VALIDATION_PERCENTAGE,
    )
    print(
        f"Total: {len(split.fine_tune)} fine-tune / {len(split.validation)} validation / "
        f"{len(split.eval)} eval instances (eval held out, unused here)"
    )

    with open(STATS_PATH) as f:
        stats = json.load(f)

    ft_dataset = EstimatorDataset(instances=split.fine_tune, stats=stats)
    val_dataset = EstimatorDataset(instances=split.validation, stats=stats)

    ft_loader = DataLoader(
        ft_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True
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

    if not ESTIMATOR_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found at {ESTIMATOR_CHECKPOINT_PATH}. "
            "Train the base estimator before fine-tuning."
        )
    model.load_state_dict(
        torch.load(ESTIMATOR_CHECKPOINT_PATH, map_location=device, weights_only=True)
    )
    print(f"Loaded pretrained weights from {ESTIMATOR_CHECKPOINT_PATH}")

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=FT_MAX_LEARNING_RATE, weight_decay=WEIGHT_DECAY
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
            model.train()
            total_train_loss = 0.0
            total_train_samples = 0

            for X, conditions, y in ft_loader:
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
                for X, conditions, y in val_loader:
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
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
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
                model.load_state_dict(best_state)
                print(f"Restored best model (val loss={best_val_loss:.5f}).")

            FINETUNED_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), FINETUNED_CHECKPOINT_PATH)
            print(f"Model saved -> {FINETUNED_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
