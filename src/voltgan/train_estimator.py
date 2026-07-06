import matplotlib

from voltgan.models.soh_estimator import SohEstimator

matplotlib.use("Agg")
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
    DATA_PATH,
    DROPOUT,
    EMBEDDING_DIM,
    ESTIMATOR_CHECKPOINT_PATH,
    FEEDFORWARD_DIM,
    INPUT_FEATURES,
    LEARNING_RATE,
    N_BLOCKS,
    N_EPOCHS,
    N_HEADS,
    RANDOM_SEED,
    TRAINING_MCUS,
    VALIDATION_MCUS,
    WINDOW_SIZE,
)
from voltgan.data import EstimatorDataset, Standardizer
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

    training_dataset = EstimatorDataset(
        mcus=TRAINING_MCUS, data_path=hdf_data_path, stats=stats
    )
    validation_dataset = EstimatorDataset(
        mcus=VALIDATION_MCUS, data_path=hdf_data_path, stats=stats
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
        max_length=WINDOW_SIZE,
    ).to(device)

    criterion = torch.nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.1
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
) -> None:
    print(
        f"Train batches: {len(training_loader)} | Validation batches: {len(validation_loader)}"
    )
    print(f"Starting training for {n_epochs} epochs...")

    for epoch in range(n_epochs):
        total_training_loss = 0.0
        total_validation_loss = 0.0

        model.train()
        for X, y in training_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            y_pred = model(X)

            loss = criterion(y_pred, y)
            loss.backward()

            total_training_loss += loss.item()

            optimizer.step()

        model.eval()
        with torch.no_grad():
            for X, y in validation_loader:
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                y_pred = model(X)

                loss = criterion(y_pred, y)
                total_validation_loss += loss.item()

        average_training_loss = total_training_loss / len(training_loader)
        average_validation_loss = total_validation_loss / len(validation_loader)

        scheduler.step(average_validation_loss)

        print(
            f"Epoch {epoch + 1:02d}/{n_epochs} | "
            f"Train Loss: {average_training_loss:.5f} | "
            f"Valid Loss: {average_validation_loss:.5f}"
        )

        if _interrupted:
            break


if __name__ == "__main__":
    main()
