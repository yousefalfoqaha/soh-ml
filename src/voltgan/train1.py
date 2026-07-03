from typing import cast

import matplotlib

matplotlib.use("Agg")
import torch
import torch._inductor.config as inductor_config
from torch.optim.lr_scheduler import LRScheduler

inductor_config.max_autotune_gemm = False
import signal

from torch.amp import GradScaler, autocast
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from voltgan.config import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    CHUNK_SIZE,
    CONDITION_DIM,
    DATA_PATH,
    DROPOUT,
    EMBEDDING_DIM,
    FEEDFORWARD_DIM,
    HIDDEN_SIZE,
    INPUT_FEATURES,
    LEARNING_RATE,
    MAX_SEQUENCE_LENGTH,
    N_BLOCKS,
    N_CONDITIONS,
    N_EPOCHS,
    N_HEADS,
    N_LAYERS,
    NOISE_DIM,
    OUTPUT_FEATURES,
    RANDOM_SEED,
    TRAINING_MCUS,
    VALIDATION_MCUS,
)
from voltgan.data import DischargeDataset, Standardizer
from voltgan.models import DiscriminatorCNN, DiscriminatorTransformer, GeneratorGru

_interrupted = False


def _handle_sigint(sig, frame):
    global _interrupted
    print("\nInterrupt received, finishing current epoch...")
    _interrupted = True


signal.signal(signal.SIGINT, _handle_sigint)


def _worker_init(worker_id):
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def collate_fn(batch):
    X_list, y_list, conditions_list = [], [], []

    for item in batch:
        X_list.append(item[0])
        conditions_list.append(item[1])
        y_list.append(item[2])

    lengths = torch.tensor([len(x) for x in X_list], dtype=torch.int64)
    max_length = max(lengths).item()

    # (batch_size, max_length)
    X_padded = pad_sequence(X_list, batch_first=True, padding_value=0.0)
    y_padded = pad_sequence(y_list, batch_first=True, padding_value=0.0)

    # (batch_size, 2)
    conditions_stacked = torch.stack(conditions_list, dim=0)

    indices = torch.arange(max_length).expand(len(lengths), max_length)

    # (batch_size, max_length)
    mask = indices < lengths.unsqueeze(1)

    # (batch_size, max_length, 1)
    mask = mask.unsqueeze(-1)

    return X_padded, conditions_stacked, y_padded, mask


def main():
    torch.manual_seed(RANDOM_SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    hdf_data_path = DATA_PATH / "hdf"
    mcus = TRAINING_MCUS + VALIDATION_MCUS

    standardizer = Standardizer(DATA_PATH)
    stats = standardizer.compute(mcus)
    standardizer.save()

    training_dataset = DischargeDataset(
        mcus=TRAINING_MCUS,
        data_path=hdf_data_path,
        stats=stats,
    )
    validation_dataset = DischargeDataset(
        mcus=VALIDATION_MCUS,
        data_path=hdf_data_path,
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
        collate_fn=collate_fn,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=_worker_init,
        collate_fn=collate_fn,
    )

    generator = GeneratorGru(
        input_features=INPUT_FEATURES,
        n_conditions=N_CONDITIONS,
        hidden_size=HIDDEN_SIZE,
        output_features=OUTPUT_FEATURES,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        noise_dim=NOISE_DIM,
    ).to(device)

    # discriminator = DiscriminatorCNN(
    #     input_features=OUTPUT_FEATURES,
    #     n_conditions=N_CONDITIONS,
    #     embedding_dim=EMBEDDING_DIM,
    #     n_blocks=N_BLOCKS,
    #     dropout=DROPOUT,
    # ).to(device)

    discriminator = DiscriminatorTransformer(
        input_features=OUTPUT_FEATURES,
        n_conditions=N_CONDITIONS,
        embedding_dim=EMBEDDING_DIM,
        n_heads=N_HEADS,
        n_blocks=N_BLOCKS,
        feedforward_dim=FEEDFORWARD_DIM,
        dropout=DROPOUT,
        max_length=MAX_SEQUENCE_LENGTH,
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
        for X_real, conditions, y_real, mask in training_loader:
            X_real = X_real.to(device, non_blocking=True)
            y_real = y_real.to(device, non_blocking=True)
            conditions = conditions.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            batch_size = y_real.size(0)
            max_length = mask.size(1)

            # stage 1: train discriminator
            discriminator_optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device_type, dtype=amp_dtype):
                y_prediction_real = discriminator(y_real, conditions, mask)
                ones = torch.ones_like(y_prediction_real, dtype=torch.float32)
                raw_loss_real = criterion(y_prediction_real, ones)
                loss_real = (raw_loss_real * mask).sum() / mask.sum()

                noise = torch.randn([batch_size, max_length, NOISE_DIM], device=device)
                y_fake, _ = generator(X_real, conditions, noise)
                y_fake = y_fake.detach()

                y_prediction_fake = discriminator(y_fake, conditions, mask)
                zeros = torch.zeros_like(y_prediction_fake, dtype=torch.float32)
                raw_loss_fake = criterion(y_prediction_fake, zeros)
                loss_fake = (raw_loss_fake * mask).sum() / mask.sum()

                discriminator_loss = loss_real + loss_fake

            scaler.scale(discriminator_loss).backward()
            scaler.step(discriminator_optimizer)

            total_discriminator_training_loss += discriminator_loss.item()

            # stage 2: train generator (truncated backprop through time, chunk size CHUNK_SIZE)
            generator_optimizer.zero_grad(set_to_none=True)
            discriminator_optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device_type, dtype=amp_dtype):
                hidden_state = None
                chunks_fake = []

                for start in range(0, max_length, CHUNK_SIZE):
                    if hidden_state is not None:
                        hidden_state = hidden_state.detach()

                    X_real_window = X_real[:, start : start + CHUNK_SIZE, :]
                    window_length = X_real_window.size(1)
                    noise = torch.randn(
                        [batch_size, window_length, NOISE_DIM], device=device
                    )

                    chunk, hidden_state = generator(
                        X_real_window, conditions, noise, hidden_state
                    )
                    chunks_fake.append(chunk)

                y_fake = torch.cat(chunks_fake, dim=1)

                y_prediction_fake = discriminator(y_fake, conditions, mask)
                ones_generator = torch.ones_like(y_prediction_fake, dtype=torch.float32)
                raw_generator_loss = criterion(y_prediction_fake, ones_generator)
                generator_loss = (raw_generator_loss * mask).sum() / mask.sum()

            scaler.scale(generator_loss).backward()
            scaler.step(generator_optimizer)
            scaler.update()

            total_generator_training_loss += generator_loss.item()

        generator_scheduler.step()
        discriminator_scheduler.step()

        generator.eval()
        discriminator.eval()
        with torch.no_grad():
            for X_real, conditions, y_real, mask in validation_loader:
                X_real = X_real.to(device, non_blocking=True)
                y_real = y_real.to(device, non_blocking=True)
                conditions = conditions.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                batch_size = y_real.size(0)
                max_length = mask.size(1)

                with autocast(device_type=device_type, dtype=amp_dtype):
                    y_prediction_real = discriminator(y_real, conditions, mask)
                    ones = torch.ones_like(y_prediction_real, dtype=torch.float32)
                    raw_validation_loss_real = criterion(y_prediction_real, ones)
                    validation_loss_real = (
                        raw_validation_loss_real * mask
                    ).sum() / mask.sum()

                    noise = torch.randn(
                        [batch_size, max_length, NOISE_DIM], device=device
                    )
                    y_fake, _ = generator(X_real, conditions, noise)

                    y_prediction_fake = discriminator(y_fake, conditions, mask)
                    zeros = torch.zeros_like(y_prediction_fake, dtype=torch.float32)
                    raw_validation_loss_fake = criterion(y_prediction_fake, zeros)
                    validation_loss_fake = (
                        raw_validation_loss_fake * mask
                    ).sum() / mask.sum()

                    ones_g_val = torch.ones_like(y_prediction_fake, dtype=torch.float32)
                    raw_validation_generator_loss = criterion(
                        y_prediction_fake, ones_g_val
                    )
                    validation_generator_loss = (
                        raw_validation_generator_loss * mask
                    ).sum() / mask.sum()

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
