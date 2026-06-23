from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from voltgan.data import McusDataset, Standardizer
from voltgan.models import BatteryEncoderTransformer
from voltgan.pipeline import HdfConvertHandler, Pipeline
from voltgan.pipeline.soh import Mf4SohHandler
from voltgan.pipeline.stats_enricher import StatsEnrichHandler

MCUS_TRAIN = ["mcu1"]
MCUS_VALID = ["mcu2"]
MCUS_TEST = ["mcu3"]

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = _PROJECT_ROOT / "dataset"
PLOTS_PATH = _PROJECT_ROOT / "plots"
RASTER_FREQ = 0.2
CHANNELS = ["U", "I", "Temp[1]", "ClimaTemp"]
RAND_SEED = 42
PLOT_EPOCHS = {1, 10, 20, 30}

N_EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.0025
WINDOW_LENGTH = 5000
STRIDE = 1000

EMBED_DIM = 64
N_HEADS = 4
FEEDFORWARD_DIM = 256
DROPOUT = 0.1


def main():
    torch.manual_seed(RAND_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    pipeline = Pipeline(
        DATA_PATH,
        handlers=[
            Mf4SohHandler(qnom=18000.0, raster=RASTER_FREQ),
            HdfConvertHandler(DATA_PATH, RASTER_FREQ, CHANNELS),
            StatsEnrichHandler(),
        ],
    )
    pipeline.run(MCUS_TRAIN + MCUS_VALID)

    hdf_data_path = DATA_PATH / "hdf"

    standardizer = Standardizer(hdf_data_path)
    stats = standardizer.compute(MCUS_TRAIN)

    dataset_train = McusDataset(
        mcus=MCUS_TRAIN,
        data_path=hdf_data_path,
        window_length=WINDOW_LENGTH,
        stride=STRIDE,
        stats=stats,
    )
    dataset_valid = McusDataset(
        mcus=MCUS_VALID,
        data_path=hdf_data_path,
        window_length=WINDOW_LENGTH,
        stride=STRIDE,
        stats=stats,
    )

    loader_train = DataLoader(
        dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = BatteryEncoderTransformer(
        embed_dim=EMBED_DIM,
        n_heads=N_HEADS,
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

    train_and_validate(
        model,
        optimizer,
        criterion,
        loader_train,
        loader_valid,
        scheduler,
        N_EPOCHS,
        device,
        stats,
    )


def train_and_validate(
    model: Module,
    optimizer: Optimizer,
    criterion: Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    scheduler,
    n_epochs: int,
    device: str,
    stats: dict,
) -> None:
    print(f"Starting training for {n_epochs} epochs on {device}...")
    print(
        f"Train batches: {len(train_loader)} | Validation batches: {len(valid_loader)}"
    )

    for epoch in range(n_epochs):
        total_valid_loss = 0.0
        total_train_loss = 0.0

        model.train()

        for X_batch, init_cond_batch, y_batch in train_loader:
            X_batch, init_cond_batch, y_batch = (
                X_batch.to(device),
                init_cond_batch.to(device),
                y_batch.to(device),
            )

            optimizer.zero_grad()
            y_pred_batch = model(X_batch, init_cond_batch)
            loss = criterion(y_pred_batch, y_batch)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        scheduler.step()

        model.eval()

        with torch.no_grad():
            plotted = False
            for X_val, init_cond_batch, y_val in valid_loader:
                X_val, init_cond_batch, y_val = (
                    X_val.to(device),
                    init_cond_batch.to(device),
                    y_val.to(device),
                )

                y_val_pred = model(X_val, init_cond_batch)
                val_loss = criterion(y_val_pred, y_val)

                total_valid_loss += val_loss.item()

                if epoch in PLOT_EPOCHS and not plotted:
                    sample_true = y_val[0, :, :].cpu().numpy()
                    sample_pred = y_val_pred[0, :, :].cpu().numpy()

                    plot_battery_comparison(sample_true, sample_pred, epoch, stats)
                    plotted = True

        mean_train_loss = total_train_loss / len(train_loader)
        mean_valid_loss = total_valid_loss / len(valid_loader)

        print(
            f"Epoch {epoch + 1:02d}/{n_epochs} | "
            f"Train Loss: {mean_train_loss:.5f} | "
            f"Valid Loss: {mean_valid_loss:.5f}"
        )


def plot_battery_comparison(
    y_true: np.ndarray, y_pred: np.ndarray, epoch: int, stats: dict
) -> None:
    u_stats = stats["U"]
    t_stats = stats["Temp[1]"]

    y_true_denorm = y_true.copy()
    y_true_denorm[:, 0] = y_true[:, 0] * u_stats["std"] + u_stats["mean"]
    y_true_denorm[:, 1] = y_true[:, 1] * t_stats["std"] + t_stats["mean"]

    y_pred_denorm = y_pred.copy()
    y_pred_denorm[:, 0] = y_pred[:, 0] * u_stats["std"] + u_stats["mean"]
    y_pred_denorm[:, 1] = y_pred[:, 1] * t_stats["std"] + t_stats["mean"]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")

    time_steps, _ = y_true.shape
    t = np.arange(time_steps)

    axs[0].plot(t, y_true_denorm[:, 0], color="black", label="True U")
    axs[0].plot(
        t, y_pred_denorm[:, 0], color="red", linestyle="--", label="Pred U", alpha=0.8
    )
    axs[0].set_title("Voltage Comparison")
    axs[0].set_ylabel("Voltage (V)")
    axs[0].set_xlabel("Time Steps (0.1s)")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    axs[1].plot(t, y_true_denorm[:, 1], color="darkred", label="True Temp")
    axs[1].plot(
        t,
        y_pred_denorm[:, 1],
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

    fig.savefig(
        PLOTS_PATH.joinpath(f"epoch_{epoch:02d}.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
