from typing import cast

import matplotlib

from voltgan.models.soh_estimator import SohEstimator

matplotlib.use("Agg")
import torch
import torch._inductor.config as inductor_config
from torch.optim.lr_scheduler import LRScheduler

inductor_config.max_autotune_gemm = False
import signal

from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from voltgan.config import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    CHUNK_SIZE,
    DATA_PATH,
    DROPOUT,
    EMBEDDING_DIM,
    FEEDFORWARD_DIM,
    INPUT_FEATURES,
    LEARNING_RATE,
    N_BLOCKS,
    N_EPOCHS,
    N_HEADS,
    RANDOM_SEED,
    TRAINING_MCUS,
    VALIDATION_MCUS,
)
from voltgan.data import DischargeDataset, Standardizer
from voltgan.models import SohEstimator

_interrupted = False


def _handle_sigint(sig, frame):
    global _interrupted
    print("\nInterrupt received, finishing current epoch...")
    _interrupted = True


signal.signal(signal.SIGINT, _handle_sigint)


def _worker_init(worker_id):
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def main():
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(RANDOM_SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    hdf_data_path = DATA_PATH / "hdf"
    mcus = TRAINING_MCUS + VALIDATION_MCUS

    standardizer = Standardizer(DATA_PATH)
    stats = standardizer.compute(mcus)
    standardizer.save()

    training_dataset = DischargeDataset(
        mcus=TRAINING_MCUS, data_path=hdf_data_path, stats=stats, windows=True
    )
    validation_dataset = DischargeDataset(
        mcus=VALIDATION_MCUS, data_path=hdf_data_path, stats=stats, windows=True
    )

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
        input_features=INPUT_FEATURES,
        embedding_dim=EMBEDDING_DIM,
        feedforward_dim=FEEDFORWARD_DIM,
        n_heads=N_HEADS,
        n_blocks=N_BLOCKS,
        dropout=DROPOUT,
    ).to(device)

    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=N_EPOCHS,
        eta_min=5e-5,
    )
    compiled_model = cast(SohEstimator, torch.compile(model))

    train_and_validate(
        compiled_model,
        optimizer,
        scheduler,
        criterion,
        training_loader,
        validation_loader,
        N_EPOCHS,
        device,
    )

    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"Model saved → {CHECKPOINT_PATH}")


def _detach_hidden_state(hidden_state):
    if hidden_state is None:
        return None
    return tuple(h.detach() for h in hidden_state)


def train_and_validate(
    model: SohEstimator,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    criterion: Module,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    n_epochs: int,
    device: str,
) -> None:
    print(
        f"Train batches: {len(training_loader)} | Validation batches: {len(validation_loader)}"
    )
    print(f"Starting training for {n_epochs} epochs...")

    for epoch in range(n_epochs):
        total_training_loss = 0.0
        train_batch_count = 0

        model.train()
        for X, y in training_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()

            optimizer.step()

        scheduler.step()

        total_validation_loss = 0.0

        model.eval()
        with torch.no_grad():
            for X, y in validation_loader:
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                y_pred = model(X)

                loss = criterion(y_pred, y)
                total_validation_loss += loss

        mean_training_loss = total_training_loss / max(1, train_batch_count)
        mean_validation_loss = total_validation_loss / max(1, val_batch_count)

        print(
            f"Epoch {epoch + 1:02d}/{n_epochs} | "
            f"Train Loss: {mean_training_loss:.5f} | "
            f"Valid Loss: {mean_validation_loss:.5f}"
        )

        if _interrupted:
            break


if __name__ == "__main__":
    main()
