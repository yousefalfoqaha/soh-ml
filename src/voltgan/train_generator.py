import signal

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from voltgan.config import (
    BATCH_SIZE,
    CONV_BASE_CHANNELS,
    CONV_HIDDEN_LAYERS,
    CONV_KERNEL_SIZE,
    CRITIC_CHECKPOINT_PATH,
    CRITIC_ITERATIONS,
    DROPOUT,
    GENERATOR_CHECKPOINT_PATH,
    GENERATOR_STATS_PATH,
    HDF_ROOT,
    LATENT_SIZE,
    LEARNING_RATE,
    LEAVE_OUT_TEMPERATURE_RANGE,
    N_CONDITIONS_GAN,
    N_EPOCHS,
    NOISE_DIM,
    RANDOM_SEED,
    TRAINING_MCUS,
    VALIDATION_MCUS,
)
from voltgan.data import BucketSampler, DischargeDataset, StatisticsCalculator
from voltgan.models import Critic, Generator
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
    train_instances = filter_by_temperature(
        train_instances, LEAVE_OUT_TEMPERATURE_RANGE, exclude=True
    )
    print(f"Training instances after temp filter: {len(train_instances)}")

    val_instances = load_instances(HDF_ROOT, VALIDATION_MCUS)
    val_instances = filter_by_temperature(
        val_instances, LEAVE_OUT_TEMPERATURE_RANGE, exclude=False
    )
    print(f"Validation instances (0C only): {len(val_instances)}")

    statistics = StatisticsCalculator(GENERATOR_STATS_PATH)
    stats = statistics.compute(train_instances)
    statistics.save()

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
        dropout=DROPOUT,
    ).to(device)

    critic = Critic(
        input_features=2,
        n_conditions=N_CONDITIONS_GAN,
        base_channels=CONV_BASE_CHANNELS,
        kernel_size=CONV_KERNEL_SIZE,
        dropout=DROPOUT,
    ).to(device)

    generator_optim = torch.optim.Adam(
        generator.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9)
    )
    critic_optim = torch.optim.Adam(
        critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9)
    )

    generator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        generator_optim, T_max=N_EPOCHS, eta_min=5e-5
    )
    critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        critic_optim, T_max=N_EPOCHS, eta_min=5e-5
    )

    print(
        f"\nTrain batches: {len(training_loader)} | "
        f"Validation batches: {len(validation_loader)}"
    )
    print(f"Training for {N_EPOCHS} epochs...")

    for epoch in range(N_EPOCHS):
        generator.train()
        critic.train()

        total_train_loss_critic = 0.0
        total_train_loss_gp = 0.0
        total_train_loss_generator = 0.0
        n_batches = 0

        for _, (X, conditions, real) in enumerate(training_loader):
            X = X.to(device, non_blocking=True)
            conditions = conditions.to(device, non_blocking=True)
            real = real.to(device, non_blocking=True)

            critic_losses = []
            gp_losses = []

            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(X.size(0), NOISE_DIM).to(device)
                fake = generator(X, conditions, noise)
                fake = fake.detach()
                critic_real = critic(real, conditions).reshape(-1)
                critic_fake = critic(fake, conditions).reshape(-1)
                gp = 10 * gradient_penalty(real, fake, critic, conditions, device)

                loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + gp
                critic.zero_grad()
                loss_critic.backward()
                critic_optim.step()

                critic_losses.append(loss_critic.item())
                gp_losses.append(gp)

            total_train_loss_critic += sum(critic_losses) / len(critic_losses)
            total_train_loss_gp += sum(gp_losses) / len(gp_losses)

            noise = torch.randn(X.size(0), NOISE_DIM).to(device)
            fake = generator(X, conditions, noise)
            output = critic(fake, conditions).reshape(-1)

            loss_generator = -torch.mean(output)

            generator.zero_grad()
            loss_generator.backward()
            generator_optim.step()

            total_train_loss_generator += loss_generator.item()
            n_batches += 1

        generator_scheduler.step()
        critic_scheduler.step()

        # generator.eval()
        # critic.eval()
        # total_validation_loss_critic = 0.0
        # total_validation_loss_generator = 0.0
        # val_batches = 0

        # with torch.no_grad():
        #     for X, conditions, y in validation_loader:
        #         X = X.to(device, non_blocking=True)
        #         conditions = conditions.to(device, non_blocking=True)
        #         y = y.to(device, non_blocking=True)
        #
        #         # WIP
        #
        #         val_batches += 1

        mean_loss_train_critic = total_train_loss_critic / max(1, n_batches)
        mean_loss_train_gp = total_train_loss_gp / max(1, n_batches)
        mean_loss_train_generator = total_train_loss_generator / max(1, n_batches)

        # mean_loss_validation_critic = total_validation_loss_critic / max(1, val_batches)
        # mean_loss_validation_generator = total_validation_loss_generator / max(
        #     1, val_batches
        # )

        print(
            f"Epoch {epoch + 1:02d}/{N_EPOCHS} | "
            f"Critic Loss: {mean_loss_train_critic:.5f} | "
            f"GP: {mean_loss_train_gp:.5f} | "
            f"Generator Loss: {mean_loss_train_generator:.5f} | "
            # f"Valid Critic Loss: {mean_loss_validation_critic:.5f}"
            # f"Valid Generator Loss: {mean_loss_validation_generator:.5f}"
        )

        if _interrupted:
            break

    torch.save(generator.state_dict(), GENERATOR_CHECKPOINT_PATH)
    torch.save(critic.state_dict(), CRITIC_CHECKPOINT_PATH)
    print(f"Model saved -> {GENERATOR_CHECKPOINT_PATH}")


def gradient_penalty(real, fake, critic, conditions, device="cpu"):
    epsilon = torch.rand(real.shape[0], 1, 1, device=device)
    interpolated_instances = epsilon * real + (1 - epsilon) * fake
    interpolated_instances = interpolated_instances.requires_grad_(True)

    mixed_scores = critic(interpolated_instances, conditions)

    gradient = torch.autograd.grad(
        inputs=interpolated_instances,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
    )[0]

    gradient = gradient.reshape(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penality = torch.mean((gradient_norm - 1) ** 2)

    return gradient_penality


if __name__ == "__main__":
    main()
