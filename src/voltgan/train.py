from pathlib import Path
from typing import cast

import matplotlib

from voltgan.models import DiscriminatorTransformer

matplotlib.use("Agg")
import torch
import torch._inductor.config as inductor_config
from torch.optim.lr_scheduler import LRScheduler

inductor_config.max_autotune_gemm = False
import signal

from torch.amp import GradScaler, autocast
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from voltgan.data import McusDataset, Standardizer
from voltgan.models import GeneratorGru
from voltgan.pipeline import (
    ChannelValidationHandler,
    ExtractDischargePeriodsHandler,
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

NOMINAL_CAPACITY = 18000.0
RASTER_FREQUENCY = 1.5
CHANNELS = ["U", "I", "Temp[1]", "ClimaTemp"]

RANDOM_SEED = 42

N_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.0005

WINDOW_LENGTH = 500
STRIDE = 500

# transformer
EMBEDDING_DIM = 128
FEEDFORWARD_DIM = 512
N_HEADS = 4
N_BLOCKS = 2
DROPOUT = 0.1

# gru
INPUT_FEATURES = 3
N_CONDITIONS = 1
HIDDEN_SIZE = 128
OUTPUT_FEATURES = 2
N_LAYERS = 2
NOISE_DIM = 32
CONDITION_DIM = 8

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
    ExtractDischargePeriodsHandler(CHANNELS),
    SohHandler(nominal_capacity=NOMINAL_CAPACITY, raster=RASTER_FREQUENCY),
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

    generator = GeneratorGru(
        n_conditions=N_CONDITIONS,
        hidden_size=HIDDEN_SIZE,
        output_features=OUTPUT_FEATURES,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        noise_dim=NOISE_DIM,
        condition_dim=CONDITION_DIM,
    ).to(device)

    discriminator = DiscriminatorTransformer(
        input_features=OUTPUT_FEATURES,
        n_conditions=N_CONDITIONS,
        embedding_dim=EMBEDDING_DIM,
        n_heads=N_HEADS,
        n_blocks=N_BLOCKS,
        feedforward_dim=FEEDFORWARD_DIM,
        dropout=DROPOUT,
        max_length=WINDOW_LENGTH,
    ).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE)
    generator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        generator_optimizer,
        T_max=N_EPOCHS,
        eta_min=1e-5,
    )
    compiled_generator = cast(GeneratorGru, torch.compile(generator))

    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=LEARNING_RATE
    )
    discriminator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        discriminator_optimizer,
        T_max=N_EPOCHS,
        eta_min=1e-5,
    )
    compiled_discriminator = cast(
        DiscriminatorTransformer, torch.compile(discriminator)
    )

    train_and_validate(
        # generator
        compiled_generator,
        generator_optimizer,
        generator_scheduler,
        # discriminator
        compiled_discriminator,
        discriminator_optimizer,
        discriminator_scheduler,
        # shared
        criterion,
        training_loader,
        validation_loader,
        scaler,
        N_EPOCHS,
        device,
    )

    torch.save(generator.state_dict(), CHECKPOINT_PATH)

    print(f"Model saved → {CHECKPOINT_PATH}")


def train_and_validate(
    generator: GeneratorGru,
    generator_optimizer: Optimizer,
    generator_scheduler: LRScheduler,
    discriminator: DiscriminatorTransformer,
    discriminator_optimizer: Optimizer,
    discriminator_scheduler: LRScheduler,
    criterion: Module,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    scaler: GradScaler,
    n_epochs: int,
    device: str,
) -> None:
    print(
        f"Train batches: {len(training_loader)} | Validation batches: {len(validation_loader)}"
    )
    print(f"Starting training for {n_epochs} epochs...")

    device_type = "cuda" if "cuda" in device else "cpu"
    amp_dtype = torch.float16 if device_type == "cuda" else torch.bfloat16

    for epoch in range(n_epochs):
        total_discriminator_training_loss = 0.0
        total_generator_training_loss = 0.0
        total_discriminator_validation_loss = 0.0
        total_generator_validation_loss = 0.0

        generator.train()
        discriminator.train()
        for conditions, y_real in training_loader:
            y_real = y_real.to(device, non_blocking=True)
            conditions = conditions.to(device, non_blocking=True)
            batch_size = y_real.size(0)
            sequence_length = y_real.size(1)

            # stage 1: train discriminator
            discriminator_optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device_type, dtype=amp_dtype):
                y_prediction_real = discriminator(y_real, conditions)
                ones = torch.ones_like(y_prediction_real, dtype=torch.float32)
                loss_real = criterion(y_prediction_real, ones)

                noise = torch.randn(
                    [batch_size, sequence_length, NOISE_DIM], device=device
                )
                y_fake, _ = generator(conditions, noise)
                y_fake = y_fake.detach()

                y_prediction_fake = discriminator(y_fake, conditions)
                zeros = torch.zeros_like(y_prediction_fake, dtype=torch.float32)
                loss_fake = criterion(y_prediction_fake, zeros)

                discriminator_loss = loss_real + loss_fake

            scaler.scale(discriminator_loss).backward()
            scaler.step(discriminator_optimizer)

            total_discriminator_training_loss += discriminator_loss.item()

            # stage 2: train generator
            generator_optimizer.zero_grad(set_to_none=True)
            discriminator_optimizer.zero_grad(set_to_none=True)

            noise = torch.randn([batch_size, sequence_length, NOISE_DIM], device=device)

            with autocast(device_type=device_type, dtype=amp_dtype):
                y_fake, _ = generator(conditions, noise)

                y_prediction_fake = discriminator(y_fake, conditions)
                ones_generator = torch.ones_like(y_prediction_fake, dtype=torch.float32)
                generator_loss = criterion(y_prediction_fake, ones_generator)

            scaler.scale(generator_loss).backward()
            scaler.step(generator_optimizer)
            scaler.update()

            total_generator_training_loss += generator_loss.item()

        generator_scheduler.step()
        discriminator_scheduler.step()

        generator.eval()
        discriminator.eval()
        with torch.no_grad():
            for conditions, y_real in validation_loader:
                y_real = y_real.to(device, non_blocking=True)
                conditions = conditions.to(device, non_blocking=True)
                batch_size = y_real.size(0)
                sequence_length = y_real.size(1)

                with autocast(device_type=device_type, dtype=amp_dtype):
                    y_prediction_real = discriminator(y_real, conditions)
                    ones = torch.ones_like(y_prediction_real, dtype=torch.float32)
                    validation_loss_real = criterion(y_prediction_real, ones)

                    noise = torch.randn(
                        [batch_size, sequence_length, NOISE_DIM], device=device
                    )
                    y_fake, _ = generator(conditions, noise)

                    y_prediction_fake = discriminator(y_fake, conditions)
                    zeros = torch.zeros_like(y_prediction_fake, dtype=torch.float32)
                    validation_loss_fake = criterion(y_prediction_fake, zeros)

                    ones_g_val = torch.ones_like(y_prediction_fake, dtype=torch.float32)
                    validation_generator_loss = criterion(y_prediction_fake, ones_g_val)

                total_discriminator_validation_loss += (
                    validation_loss_real + validation_loss_fake
                ).item()
                total_generator_validation_loss += validation_generator_loss.item()

        mean_discriminator_training = total_discriminator_training_loss / len(
            training_loader
        )
        mean_generator_training = total_generator_training_loss / len(training_loader)
        mean_discriminator_validation = total_discriminator_validation_loss / len(
            validation_loader
        )
        mean_generator_validation = total_generator_validation_loss / len(
            validation_loader
        )

        print(
            f"Epoch {epoch + 1:02d}/{n_epochs} | "
            f"Train D Loss: {mean_discriminator_training:.5f} | Train G Loss: {mean_generator_training:.5f} || "
            f"Valid D Loss: {mean_discriminator_validation:.5f} | Valid G Loss: {mean_generator_validation:.5f}"
        )

        if _interrupted:
            break


if __name__ == "__main__":
    main()
