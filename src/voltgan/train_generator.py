import signal

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from voltgan.config import (
    AUTOENCODER_EPOCHS,
    BATCH_SIZE,
    CONV_BASE_CHANNELS,
    CONV_CHANNEL_MULTS,
    CONV_KERNEL_SIZE,
    GENERATOR_CHECKPOINT_PATH,
    GENERATOR_STATS_PATH,
    HDF_ROOT,
    LATENT_DIM,
    LATENT_LENGTH,
    LEARNING_RATE,
    LEAVE_OUT_TEMPERATURE_RANGE,
    PADDED_LENGTH,
    RANDOM_SEED,
    TRAINING_MCUS,
    VALIDATION_MCUS,
)
from voltgan.data import DischargeDataset, Standardizer
from voltgan.models import BatterySequenceGenerator
from voltgan.utils.discover import filter_by_temperature, load_instances

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

    lengths = torch.tensor([len(x) for x in X_list], dtype=torch.int64)

    X_padded = pad_sequence(X_list, batch_first=True, padding_value=0.0)
    y_padded = pad_sequence(y_list, batch_first=True, padding_value=0.0)

    if X_padded.size(1) < PADDED_LENGTH:
        pad_len = PADDED_LENGTH - X_padded.size(1)
        X_padded = F.pad(X_padded, (0, 0, 0, pad_len))
        y_padded = F.pad(y_padded, (0, 0, 0, pad_len))

    conditions_stacked = torch.stack(conditions_list, dim=0)

    indices = torch.arange(PADDED_LENGTH).expand(len(lengths), PADDED_LENGTH)
    mask = indices < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1)

    return X_padded, conditions_stacked, y_padded, mask


def main():
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(RANDOM_SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_instances = load_instances(HDF_ROOT, TRAINING_MCUS)
    train_instances = filter_by_temperature(
        train_instances, LEAVE_OUT_TEMPERATURE_RANGE, exclude=True
    )
    print(f"Training instances after temp filter: {len(train_instances)}")

    val_instances = load_instances(HDF_ROOT, VALIDATION_MCUS)
    val_instances = filter_by_temperature(
        val_instances, LEAVE_OUT_TEMPERATURE_RANGE, exclude=False
    )
    print(f"Validation instances (0C only): {len(val_instances)}")

    standardizer = Standardizer(GENERATOR_STATS_PATH)
    stats = standardizer.compute(train_instances)
    standardizer.save()

    training_dataset = DischargeDataset(instances=train_instances, stats=stats)
    validation_dataset = DischargeDataset(instances=val_instances, stats=stats)

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

    model = BatterySequenceGenerator(
        padded_length=PADDED_LENGTH,
        latent_length=LATENT_LENGTH,
        latent_dim=LATENT_DIM,
        conv_base_channels=CONV_BASE_CHANNELS,
        conv_channel_mults=CONV_CHANNEL_MULTS,
        conv_kernel_size=CONV_KERNEL_SIZE,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=AUTOENCODER_EPOCHS, eta_min=5e-5
    )

    print(
        f"\nTrain batches: {len(training_loader)} | "
        f"Validation batches: {len(validation_loader)}"
    )
    print(f"Training Conv Autoencoder for {AUTOENCODER_EPOCHS} epochs...")

    for epoch in range(AUTOENCODER_EPOCHS):
        model.train()
        total_loss = 0.0
        total_v_loss = 0.0
        total_t_loss = 0.0
        n_batches = 0

        for _, _, y, mask in training_loader:
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            y_pred = model(y)
            diff = (y_pred - y) ** 2
            loss = (diff * mask).sum() / (mask.sum() + 1e-8)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_v_loss += (diff[:, :, 0:1] * mask).sum().item() / (
                mask.sum().item() + 1e-8
            )
            total_t_loss += (diff[:, :, 1:2] * mask).sum().item() / (
                mask.sum().item() + 1e-8
            )
            n_batches += 1

        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for _, _, y, mask in validation_loader:
                y = y.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                y_pred = model(y)
                diff = (y_pred - y) ** 2
                v_loss = (diff * mask).sum() / (mask.sum() + 1e-8)
                val_loss += v_loss.item()
                val_batches += 1

        mean_train = total_loss / max(1, n_batches)
        mean_train_v = total_v_loss / max(1, n_batches)
        mean_train_t = total_t_loss / max(1, n_batches)
        mean_val = val_loss / max(1, val_batches)

        print(
            f"Epoch {epoch + 1:02d}/{AUTOENCODER_EPOCHS} | "
            f"Train: {mean_train:.5f} (V={mean_train_v:.5f} T={mean_train_t:.5f}) | "
            f"Valid: {mean_val:.5f}"
        )

        if _interrupted:
            break

    torch.save(model.state_dict(), GENERATOR_CHECKPOINT_PATH)
    print(f"Model saved -> {GENERATOR_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()

