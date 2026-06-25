from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch._inductor.config as inductor_config

inductor_config.max_autotune_gemm = False
import signal

from torch.amp import GradScaler, autocast
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from voltgan.data import McusDataset, Standardizer
from voltgan.models import BatteryEncoderTransformer
from voltgan.pipeline import (
    ChannelValidationHandler,
    HdfConvertHandler,
    Pipeline,
    SohHandler,
    StatsEnrichHandler,
)

TRAINING_MCUS = ["mcu1"]
VALIDATION_MCUS = ["mcu2"]
TESTING_MCUS = ["mcu3"]

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = _PROJECT_ROOT / "dataset"
PLOTS_PATH = _PROJECT_ROOT / "plots"
CHECKPOINT_PATH = _PROJECT_ROOT / "model.pt"

RASTER_FREQUENCY = 0.1
CHANNELS = ["U", "I", "Temp[1]", "ClimaTemp"]
PLOT_EPOCHS = {1, 10, 20, 30}

RANDOM_SEED = 42

N_EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.0025

WINDOW_LENGTH = 8000
STRIDE = 4000

EMBEDDING_DIM = 128
FEEDFORWARD_DIM = 512
N_HEADS = 8
N_BLOCKS = 2
DROPOUT = 0.1

_interrupted = False


def _handle_sigint(sig, frame):
    global _interrupted
    print("\nInterrupt received, finishing current epoch...")
    _interrupted = True


signal.signal(signal.SIGINT, _handle_sigint)


def _worker_init(worker_id):
    signal.signal(signal.SIGINT, signal.SIG_IGN)


_PIPELINE_HANDLERS = [
    ChannelValidationHandler(CHANNELS),
    SohHandler(nominal_charge=18000.0, raster=RASTER_FREQUENCY),
    HdfConvertHandler(DATA_PATH, RASTER_FREQUENCY, CHANNELS),
    StatsEnrichHandler(),
]


def main():
    torch.manual_seed(RANDOM_SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    pipeline = Pipeline(DATA_PATH, _PIPELINE_HANDLERS)
    pipeline.run(TRAINING_MCUS + VALIDATION_MCUS)

    hdf_data_path = DATA_PATH / "hdf"

    standardizer = Standardizer(DATA_PATH)
    stats = standardizer.compute(TRAINING_MCUS)
    standardizer.save(stats)

    training_dataset = McusDataset(
        mcus=TRAINING_MCUS,
        data_path=hdf_data_path,
        window_length=WINDOW_LENGTH,
        stride=STRIDE,
        stats=stats,
    )
    validation_dataset = McusDataset(
        mcus=VALIDATION_MCUS,
        data_path=hdf_data_path,
        window_length=WINDOW_LENGTH,
        stride=STRIDE,
        stats=stats,
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

    model = BatteryEncoderTransformer(
        embedding_dim=EMBEDDING_DIM,
        n_heads=N_HEADS,
        n_blocks=N_BLOCKS,
        window_length=WINDOW_LENGTH,
        feedforward_dim=FEEDFORWARD_DIM,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.HuberLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=N_EPOCHS,
        eta_min=1e-5,
    )
    scaler = GradScaler()
    compiled_model = cast(BatteryEncoderTransformer, torch.compile(model))

    train_and_validate(
        compiled_model,
        optimizer,
        criterion,
        training_loader,
        validation_loader,
        scheduler,
        scaler,
        N_EPOCHS,
        device,
        stats,
    )

    torch.save(model.state_dict(), CHECKPOINT_PATH)

    print(f"Model saved → {CHECKPOINT_PATH}")


def train_and_validate(
    model: Module,
    optimizer: Optimizer,
    criterion: Module,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    scheduler,
    scaler: GradScaler,
    n_epochs: int,
    device: str,
    stats: dict,
) -> None:
    print(
        f"Train batches: {len(training_loader)} | Validation batches: {len(validation_loader)}"
    )
    print(f"Starting training for {n_epochs} epochs...")

    for epoch in range(n_epochs):
        total_training_loss = 0.0
        total_validation_loss = 0.0

        model.train()
        for X, initial_conditions, y in training_loader:
            X = X.to(device, non_blocking=True)
            initial_conditions = initial_conditions.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast(device_type="cuda"):
                y_prediction = model(X, initial_conditions)
                loss = criterion(y_prediction, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            total_training_loss += loss.item()

        scheduler.step()

        model.eval()
        plotted = False
        with torch.no_grad():
            for X, initial_conditions, y in validation_loader:
                X = X.to(device, non_blocking=True)
                initial_conditions = initial_conditions.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with autocast(device_type="cuda"):
                    y_prediction = model(X, initial_conditions)
                    loss = criterion(y_prediction, y)

                total_validation_loss += loss.item()

                if epoch in PLOT_EPOCHS and not plotted:
                    plot_battery_comparison(
                        y[0, :, :].cpu().numpy(),
                        y_prediction[0, :, :].cpu().numpy(),
                        epoch,
                        stats,
                    )
                    plotted = True

        mean_training_loss = total_training_loss / len(training_loader)
        mean_validation_loss = total_validation_loss / len(validation_loader)

        print(
            f"Epoch {epoch + 1:02d}/{n_epochs} | "
            f"Train Loss: {mean_training_loss:.5f} | "
            f"Valid Loss: {mean_validation_loss:.5f}"
        )

        if _interrupted:
            break


def plot_battery_comparison(
    y_true: np.ndarray, y_pred: np.ndarray, epoch: int, stats: dict
) -> None:
    u_stats = stats["U"]
    t_stats = stats["Temp[1]"]

    y_true_denorm = y_true.copy()
    y_true_denorm[:, 0] = y_true[:, 0] * u_stats["standard_deviation"] + u_stats["mean"]
    y_true_denorm[:, 1] = y_true[:, 1] * t_stats["standard_deviation"] + t_stats["mean"]

    y_prediction_denorm = y_pred.copy()
    y_prediction_denorm[:, 0] = (
        y_pred[:, 0] * u_stats["standard_deviation"] + u_stats["mean"]
    )
    y_prediction_denorm[:, 1] = (
        y_pred[:, 1] * t_stats["standard_deviation"] + t_stats["mean"]
    )

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    timestamps = np.arange(y_true.shape[0])

    axs[0].plot(timestamps, y_true_denorm[:, 0], color="black", label="True U")
    axs[0].plot(
        timestamps,
        y_prediction_denorm[:, 0],
        color="red",
        linestyle="--",
        label="Pred U",
        alpha=0.8,
    )
    axs[0].set_title("Voltage Comparison")
    axs[0].set_ylabel("Voltage (V)")
    axs[0].set_xlabel("Time Steps (0.1s)")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    axs[1].plot(timestamps, y_true_denorm[:, 1], color="darkred", label="True Temp")
    axs[1].plot(
        timestamps,
        y_prediction_denorm[:, 1],
        color="blue",
        linestyle="--",
        label="Pred Temp",
        alpha=0.8,
    )
    axs[1].set_title("Temperature Comparison")
    axs[1].set_ylabel("Temperature (°C)")
    axs[1].set_xlabel("Time Steps (0.1s)")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    fig.suptitle(f"Epoch {epoch} Results")
    PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_PATH / f"epoch_{epoch:02d}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
