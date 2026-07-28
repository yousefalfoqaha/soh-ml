from __future__ import annotations

import signal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from voltgan.config import CHUNK_SIZE, NOISE_DIM
from voltgan.models import GeneratorClient

_interrupted = False


def _handle_sigint(sig, frame):
    global _interrupted
    print("\nInterrupt received, finishing current epoch...")
    _interrupted = True


signal.signal(signal.SIGINT, _handle_sigint)


def _worker_init(worker_id):
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def main() -> None:
    import torch.nn.functional as F
    from torch.nn.utils.rnn import pad_sequence

    from voltgan.config import (
        BATCH_SIZE,
        CONV_HIDDEN_LAYERS,
        GENERATOR_CHECKPOINT_PATH,
        GENERATOR_STATS_PATH,
        HDF_ROOT,
        LEARNING_RATE,
        N_EPOCHS,
        RANDOM_SEED,
        TRAINING_MCUS,
        VALIDATION_MCUS,
    )
    from voltgan.dataset import (
        BucketSampler,
        DischargeDataset,
        InstanceRepository,
        StatisticsCalculator,
    )

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(RANDOM_SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    repo = InstanceRepository(root=HDF_ROOT)
    train_instances = repo.load(TRAINING_MCUS)
    print(f"Training instances: {len(train_instances)}")

    val_instances = repo.load(VALIDATION_MCUS)
    print(f"Validation instances: {len(val_instances)}")

    standardizer = StatisticsCalculator(GENERATOR_STATS_PATH)
    stats = standardizer.compute(train_instances)
    standardizer.save()

    training_dataset = DischargeDataset(instances=train_instances, stats=stats)
    validation_dataset = DischargeDataset(instances=val_instances, stats=stats)

    def collate_fn(batch):
        X_list, cond_list, y_list = [], [], []
        for item in batch:
            X_list.append(item[0])
            cond_list.append(item[1])
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
        conditions_stacked = torch.stack(cond_list, dim=0)
        return X_padded, conditions_stacked, y_padded

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

    client = GeneratorClient(device=device, checkpoint_path=None, is_training=True)
    optimizer = torch.optim.Adam(client.model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss(reduction="sum")

    print(
        f"\nTrain batches: {len(training_loader)} | "
        f"Validation batches: {len(validation_loader)}"
    )
    print(f"Training for {N_EPOCHS} epochs...")

    global _interrupted

    for epoch in range(N_EPOCHS):
        client.train()
        total_train_loss = 0.0
        n_batches = 0

        for X, conditions, y in training_loader:
            X = X.to(device, non_blocking=True)
            conditions = conditions.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            noise = torch.rand(X.size(0), NOISE_DIM, device=device)
            seq_len = X.size(1)
            optimizer.zero_grad(set_to_none=True)
            h = None
            batch_loss = 0.0
            for start in range(0, seq_len, CHUNK_SIZE):
                end = start + CHUNK_SIZE
                X_c = X[:, start:end, :]
                y_c = y[:, start:end, :]

                latent_input = client.model.encode(X_c, conditions, noise)
                out, h = client.model.gru(latent_input, h)
                y_pred_chunk = client.model.output(out)
                chunk_loss = criterion(y_pred_chunk, y_c) / y.numel()
                chunk_loss.backward()
                batch_loss += chunk_loss.item()
                h = h.detach()

            optimizer.step()
            total_train_loss += batch_loss
            n_batches += 1

        client.eval()
        total_val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for X, conditions, y in validation_loader:
                X = X.to(device, non_blocking=True)
                conditions = conditions.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                noise = torch.rand(X.size(0), NOISE_DIM, device=device)
                y_pred = client(X, conditions, noise)
                loss = criterion(y_pred, y) / y.numel()
                total_val_loss += loss.item()
                n_val_batches += 1

        mean_train = total_train_loss / max(1, n_batches)
        if n_val_batches > 0:
            mean_val = total_val_loss / n_val_batches
            val_note = f"{mean_val:.5f}"
        else:
            print("  (no validation batches; using train loss for scheduler)")
            mean_val = mean_train
            val_note = "n/a (using train)"

        cur_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:02d}/{N_EPOCHS} | "
            f"Train Loss: {mean_train:.5f} | "
            f"Valid Loss: {val_note} | "
            f"LR: {cur_lr:.2e}"
        )

        if _interrupted:
            break

    torch.save(client.model.state_dict(), GENERATOR_CHECKPOINT_PATH)
    print(f"Model saved -> {GENERATOR_CHECKPOINT_PATH}")

