import matplotlib

from voltgan.models.soh_estimator import SohEstimator

matplotlib.use("Agg")
import json

import torch
import torch._inductor.config as inductor_config
from torch.optim.lr_scheduler import ReduceLROnPlateau

inductor_config.max_autotune_gemm = False
import signal

from torch.nn import Module
from torch.optim import Optimizer
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
    HDF_ROOT,
    LEARNING_RATE,
    N_EPOCHS,
    RANDOM_SEED,
    STATS_PATH,
    TESTING_MCUS,
    TRAINING_MCUS,
    VALIDATION_MCUS,
)
from voltgan.data import EstimatorDataset
from voltgan.models import SohEstimator
from voltgan.utils.discover import load_instances

_interrupted = False


def _handle_sigint(sig, frame):
    global _interrupted
    print("\nInterrupt received, finishing current epoch...")
    _interrupted = True


signal.signal(signal.SIGINT, _handle_sigint)


def _worker_init(worker_id):
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def main():
    torch.manual_seed(RANDOM_SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    hdf_data_path = HDF_ROOT

    train_instances = load_instances(hdf_data_path, TRAINING_MCUS)
    val_mcus = VALIDATION_MCUS + TESTING_MCUS
    val_instances = load_instances(hdf_data_path, val_mcus)

    with open(STATS_PATH) as f:
        stats = json.load(f)

    training_dataset = EstimatorDataset(instances=train_instances, stats=stats)
    validation_dataset = EstimatorDataset(instances=val_instances, stats=stats)

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

    model = SohEstimator(
        input_features=ESTIMATOR_INPUT_FEATURES,
        n_conditions=ESTIMATOR_N_CONDITIONS,
        base_channels=ESTIMATOR_BASE_CHANNELS,
        stride=ESTIMATOR_STRIDE,
        kernel_size=ESTIMATOR_KERNEL_SIZE,
        gru_hidden_size=ESTIMATOR_GRU_HIDDEN_SIZE,
        gru_n_layers=ESTIMATOR_GRU_N_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.05
    )

    train_and_validate(
        model,
        optimizer,
        scheduler,
        criterion,
        training_loader,
        validation_loader,
        N_EPOCHS,
        device,
    )

    torch.save(model.state_dict(), ESTIMATOR_CHECKPOINT_PATH)
    print(f"Model saved → {ESTIMATOR_CHECKPOINT_PATH}")


def train_and_validate(
    model: SohEstimator,
    optimizer: Optimizer,
    scheduler: ReduceLROnPlateau,
    criterion: Module,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    n_epochs: int,
    device: str,
    patience: int = 50,
) -> None:
    print(
        f"Train batches: {len(training_loader)} | Validation batches: {len(validation_loader)}"
    )
    print(
        f"Starting training for {n_epochs} epochs (early stopping patience={patience})..."
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    prev_lr = optimizer.param_groups[0]["lr"]

    for epoch in range(n_epochs):
        total_training_loss = 0.0
        total_validation_loss = 0.0

        model.train()
        for X, conditions, y in training_loader:
            X = X.to(device, non_blocking=True)
            conditions = conditions.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            y_pred = model(X, conditions)

            loss = criterion(y_pred, y)
            loss.backward()

            total_training_loss += loss.item()

            optimizer.step()

        model.eval()
        with torch.no_grad():
            for X, conditions, y in validation_loader:
                X = X.to(device, non_blocking=True)
                conditions = conditions.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                y_pred = model(X, conditions)

                loss = criterion(y_pred, y)
                total_validation_loss += loss.item()

        average_training_loss = total_training_loss / len(training_loader)
        average_validation_loss = total_validation_loss / len(validation_loader)

        scheduler.step(average_validation_loss)

        cur_lr = optimizer.param_groups[0]["lr"]
        lr_marker = " ↓" if cur_lr < prev_lr - 1e-12 else ""
        prev_lr = cur_lr

        print(
            f"Epoch {epoch + 1:02d}/{n_epochs} | "
            f"lr={cur_lr:.2e}{lr_marker} | "
            f"Train Loss: {average_training_loss:.5f} | "
            f"Valid Loss: {average_validation_loss:.5f}"
        )

        if average_validation_loss < best_val_loss:
            best_val_loss = average_validation_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)."
                )
                break

        if _interrupted:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model (val loss={best_val_loss:.5f}).")


if __name__ == "__main__":
    main()
