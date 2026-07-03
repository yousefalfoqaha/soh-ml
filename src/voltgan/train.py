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
    DATA_PATH,
    DROPOUT,
    HIDDEN_SIZE,
    INPUT_FEATURES,
    LEARNING_RATE,
    N_CONDITIONS,
    N_EPOCHS,
    N_LAYERS,
    OUTPUT_FEATURES,
    RANDOM_SEED,
    TRAINING_MCUS,
    VALIDATION_MCUS,
)
from voltgan.data import DischargeDataset, Standardizer
from voltgan.models import GeneratorGru

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

    model = GeneratorGru(
        output_features=OUTPUT_FEATURES,
        input_features=INPUT_FEATURES,
        n_conditions=N_CONDITIONS,
        hidden_size=HIDDEN_SIZE,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    criterion = torch.nn.HuberLoss(reduction="none")
    scaler = GradScaler()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=N_EPOCHS,
        eta_min=5e-5,
    )
    compiled_model = cast(GeneratorGru, torch.compile(model))

    train_and_validate(
        compiled_model,
        optimizer,
        scheduler,
        criterion,
        training_loader,
        validation_loader,
        scaler,
        N_EPOCHS,
        device,
    )

    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"Model saved → {CHECKPOINT_PATH}")


def train_and_validate(
    model: GeneratorGru,
    optimizer: Optimizer,
    scheduler: LRScheduler,
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

    # Loss weights: [Voltage Weight, Temperature Weight]
    # We multiply temperature error by 10 so it isn't drowned out by voltage spikes
    loss_weights = torch.tensor([1.0, 10.0], device=device, dtype=torch.float32).view(
        1, 1, 2
    )

    for epoch in range(n_epochs):
        total_training_loss = 0.0
        train_chunk_count = 0

        model.train()
        for X_real, conditions, y_real, mask in training_loader:
            X_real = X_real.to(device, non_blocking=True)
            y_real = y_real.to(device, non_blocking=True)
            conditions = conditions.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            max_length = mask.size(1)
            num_chunks = (max_length + CHUNK_SIZE - 1) // CHUNK_SIZE

            hidden_state = None

            # 1. MOVED ZERO_GRAD OUTSIDE THE LOOP
            optimizer.zero_grad(set_to_none=True)

            for start in range(0, max_length, CHUNK_SIZE):
                if hidden_state is not None:
                    hidden_state = hidden_state.detach()

                X_chunk = X_real[:, start : start + CHUNK_SIZE, :]
                y_chunk = y_real[:, start : start + CHUNK_SIZE, :]
                mask_chunk = mask[:, start : start + CHUNK_SIZE, :]

                with autocast(device_type=device_type, dtype=amp_dtype):
                    y_pred_chunk, hidden_state = model(
                        X_chunk, conditions, hidden_state
                    )

                    raw_loss = criterion(y_pred_chunk, y_chunk)

                    # 2. APPLY WEIGHTED LOSS
                    weighted_loss = raw_loss * loss_weights

                    chunk_loss = (weighted_loss * mask_chunk).sum() / (
                        mask_chunk.sum() * 2 + 1e-8
                    )

                # 3. ACCUMULATE GRADIENTS (scaled by num_chunks to prevent explosion)
                scaler.scale(chunk_loss / num_chunks).backward()

                total_training_loss += chunk_loss.item()
                train_chunk_count += 1

            # 4. STEP OUTSIDE THE CHUNK LOOP
            # Unscale and clip gradients to prevent sudden spikes
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        total_validation_loss = 0.0
        val_batch_count = 0

        model.eval()
        with torch.no_grad():
            for X_real, conditions, y_real, mask in validation_loader:
                X_real = X_real.to(device, non_blocking=True)
                y_real = y_real.to(device, non_blocking=True)
                conditions = conditions.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                with autocast(device_type=device_type, dtype=amp_dtype):
                    y_pred, _ = model(X_real, conditions)

                    raw_val_loss = criterion(y_pred, y_real)
                    # Apply the same weights to validation so the metrics align
                    weighted_val_loss = raw_val_loss * loss_weights
                    val_loss = (weighted_val_loss * mask).sum() / (
                        mask.sum() * 2 + 1e-8
                    )

                total_validation_loss += val_loss.item()
                val_batch_count += 1

        mean_training_loss = total_training_loss / max(1, train_chunk_count)
        mean_validation_loss = total_validation_loss / max(1, val_batch_count)

        print(
            f"Epoch {epoch + 1:02d}/{n_epochs} | "
            f"Train MSE: {mean_training_loss:.5f} | "
            f"Valid MSE: {mean_validation_loss:.5f}"
        )

        if _interrupted:
            break


if __name__ == "__main__":
    main()
