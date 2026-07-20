import signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from voltgan.config import (
    BATCH_SIZE,
    CONV_BASE_CHANNELS,
    CONV_HIDDEN_LAYERS,
    CONV_KERNEL_SIZE,
    GENERATOR_CHECKPOINT_PATH,
    GENERATOR_STATS_PATH,
    HDF_ROOT,
    LATENT_SIZE,
    LEARNING_RATE,
    N_CONDITIONS_GAN,
    N_EPOCHS_GAN,
    NOISE_DIM,
    RANDOM_SEED,
    TRAINING_MCUS,
    VALIDATION_MCUS,
)
from voltgan.data import BucketSampler, DischargeDataset, Standardizer
from voltgan.models import Generator
from voltgan.utils.discover import load_instances

_interrupted = False


def _handle_sigint(sig, frame):
    global _interrupted
    print("\nInterrupt received, finishing current epoch...")
    _interrupted = True


signal.signal(signal.SIGINT, _handle_sigint)


def _worker_init(worker_id):
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def collate_fn(batch):
    X_list, conditions_list, y_list = [], [], []

    for item in batch:
        X_list.append(item[0])
        conditions_list.append(item[1])
        y_list.append(item[2])

    X_padded = pad_sequence(X_list, batch_first=True, padding_value=0.0)
    y_padded = pad_sequence(y_list, batch_first=True, padding_value=0.0)

    max_len = X_padded.size(1)
    downsample_factor = 5**CONV_HIDDEN_LAYERS
    remainder = max_len % downsample_factor

    if remainder != 0:
        pad_len = downsample_factor - remainder
        X_padded = F.pad(X_padded, (0, 0, 0, pad_len), value=0.0)
        y_padded = F.pad(y_padded, (0, 0, 0, pad_len), value=0.0)

    conditions_stacked = torch.stack(conditions_list, dim=0)

    return X_padded, conditions_stacked, y_padded


def main():
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(RANDOM_SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_instances = load_instances(HDF_ROOT, TRAINING_MCUS)
    print(f"Training instances: {len(train_instances)}")

    val_instances = load_instances(HDF_ROOT, VALIDATION_MCUS)
    print(f"Validation instances: {len(val_instances)}")

    standardizer = Standardizer(GENERATOR_STATS_PATH)
    stats = standardizer.compute(train_instances)
    standardizer.save()

    training_dataset = DischargeDataset(instances=train_instances, stats=stats)
    validation_dataset = DischargeDataset(instances=val_instances, stats=stats)

    batch_sampler = BucketSampler(
        dataset=training_dataset,
        max_batch_size=BATCH_SIZE,
        max_padding_threshold=150,
        noise_scale=50,
        min_length=1000,
    )
    validation_sampler = BucketSampler(
        dataset=validation_dataset,
        max_batch_size=BATCH_SIZE,
        max_padding_threshold=150,
        noise_scale=50,
        min_length=1000,
    )

    training_loader = DataLoader(
        training_dataset,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=_worker_init,
        collate_fn=collate_fn,
        batch_sampler=batch_sampler,
    )
    validation_loader = DataLoader(
        validation_dataset,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=_worker_init,
        collate_fn=collate_fn,
        batch_sampler=validation_sampler,
    )

    generator = Generator(
        input_features=1,
        n_conditions=N_CONDITIONS_GAN,
        base_channels=CONV_BASE_CHANNELS,
        kernel_size=CONV_KERNEL_SIZE,
        noise_dim=NOISE_DIM,
        latent_size=LATENT_SIZE,
    ).to(device)

    optimizer = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    print(
        f"\nTrain batches: {len(training_loader)} | "
        f"Validation batches: {len(validation_loader)}"
    )
    print(f"Training for {N_EPOCHS_GAN} epochs...")

    for epoch in range(N_EPOCHS_GAN):
        generator.train()
        total_train_loss = 0.0
        n_batches = 0

        for X, conditions, y in training_loader:
            X = X.to(device, non_blocking=True)
            conditions = conditions.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            noise = torch.rand(X.size(0), NOISE_DIM, device=device)
            y_pred = generator(X, conditions, noise)

            loss = criterion(y_pred, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            n_batches += 1

        generator.eval()
        total_val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for X, conditions, y in validation_loader:
                X = X.to(device, non_blocking=True)
                conditions = conditions.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                noise = torch.rand(X.size(0), NOISE_DIM, device=device)
                y_pred = generator(X, conditions, noise)

                loss = criterion(y_pred, y)
                total_val_loss += loss.item()
                n_val_batches += 1

        mean_train_loss = total_train_loss / max(1, n_batches)

        if n_val_batches > 0:
            mean_val_loss = total_val_loss / n_val_batches
            val_note = f"{mean_val_loss:.5f}"
        else:
            print("  (no validation batches; using train loss for scheduler)")
            mean_val_loss = mean_train_loss
            val_note = "n/a (using train)"

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:02d}/{N_EPOCHS_GAN} | "
            f"Train Loss: {mean_train_loss:.5f} | "
            f"Valid Loss: {val_note} | "
            f"LR: {current_lr:.2e}"
        )

        if _interrupted:
            break

    torch.save(generator.state_dict(), GENERATOR_CHECKPOINT_PATH)
    print(f"Model saved -> {GENERATOR_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
