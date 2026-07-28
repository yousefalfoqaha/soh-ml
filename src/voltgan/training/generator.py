from __future__ import annotations

import signal

import torch
from torch.utils.data import DataLoader

from voltgan.config import CRITIC_ITERATIONS, NOISE_DIM
from voltgan.models import Critic, GeneratorClient

_interrupted = False


def _handle_sigint(sig, frame):
    global _interrupted
    print("\nInterrupt received, finishing current epoch...")
    _interrupted = True


signal.signal(signal.SIGINT, _handle_sigint)


def _worker_init(worker_id):
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def gradient_penalty(real, fake, critic, conditions, device="cpu"):
    epsilon = torch.rand(real.shape[0], 1, 1, device=device)
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated = interpolated.requires_grad_(True)

    mixed_scores = critic(interpolated, conditions)
    gradient = torch.autograd.grad(
        inputs=interpolated,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
    )[0]
    gradient = gradient.reshape(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    return torch.mean((gradient_norm - 1) ** 2)


def main() -> None:
    import torch.nn.functional as F
    from torch.nn.utils.rnn import pad_sequence

    from voltgan.config import (
        BATCH_SIZE,
        CONV_BASE_CHANNELS,
        CONV_HIDDEN_LAYERS,
        CONV_KERNEL_SIZE,
        CRITIC_CHECKPOINT_PATH,
        DROPOUT,
        GENERATOR_CHECKPOINT_PATH,
        GENERATOR_STATS_PATH,
        HDF_ROOT,
        LEARNING_RATE,
        LEAVE_OUT_TEMPERATURE_RANGE,
        N_CONDITIONS_GAN,
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

    t_lo, t_hi = LEAVE_OUT_TEMPERATURE_RANGE
    repo = InstanceRepository(root=HDF_ROOT)
    train_instances = [
        i
        for i in repo.load(TRAINING_MCUS)
        if not (t_lo <= i.ambient_temperature <= t_hi)
    ]
    print(f"Training instances after temp filter: {len(train_instances)}")

    val_instances = [
        i for i in repo.load(VALIDATION_MCUS) if t_lo <= i.ambient_temperature <= t_hi
    ]
    print(f"Validation instances (0C only): {len(val_instances)}")

    statistics = StatisticsCalculator(GENERATOR_STATS_PATH)
    stats = statistics.compute(train_instances)
    statistics.save()

    training_dataset = DischargeDataset(instances=train_instances, stats=stats)

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
    training_loader = DataLoader(
        training_dataset,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=_worker_init,
        collate_fn=collate_fn,
        batch_sampler=batch_sampler,
    )

    generator = GeneratorClient(device=device, checkpoint_path=None, is_training=True)
    critic = Critic(
        input_features=2,
        n_conditions=N_CONDITIONS_GAN,
        base_channels=CONV_BASE_CHANNELS,
        kernel_size=CONV_KERNEL_SIZE,
        dropout=DROPOUT,
    ).to(device)

    gen_optim = torch.optim.Adam(
        generator.model.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9)
    )
    crit_optim = torch.optim.Adam(
        critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9)
    )
    gen_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        gen_optim, T_max=N_EPOCHS, eta_min=5e-5
    )
    crit_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        crit_optim, T_max=N_EPOCHS, eta_min=5e-5
    )

    print(f"\nTrain batches: {len(training_loader)}")
    print(f"Training for {N_EPOCHS} epochs...")

    global _interrupted

    for epoch in range(N_EPOCHS):
        generator.train()
        critic.train()

        total_crit = 0.0
        total_gp = 0.0
        total_gen = 0.0
        n_batches = 0

        for _, (X, conditions, real) in enumerate(training_loader):
            X = X.to(device, non_blocking=True)
            conditions = conditions.to(device, non_blocking=True)
            real = real.to(device, non_blocking=True)

            crit_losses = []
            gp_losses = []

            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(X.size(0), NOISE_DIM).to(device)
                fake = generator(X, conditions, noise).detach()
                critic_real = critic(real, conditions).reshape(-1)
                critic_fake = critic(fake, conditions).reshape(-1)
                gp = 10 * gradient_penalty(real, fake, critic, conditions, device)
                loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + gp
                crit_optim.zero_grad()
                loss_critic.backward()
                crit_optim.step()
                crit_losses.append(loss_critic.item())
                gp_losses.append(gp)

            total_crit += sum(crit_losses) / len(crit_losses)
            total_gp += sum(gp_losses) / len(gp_losses)

            noise = torch.randn(X.size(0), NOISE_DIM).to(device)
            fake = generator(X, conditions, noise)
            output = critic(fake, conditions).reshape(-1)
            loss_gen = -torch.mean(output)
            gen_optim.zero_grad()
            loss_gen.backward()
            gen_optim.step()

            total_gen += loss_gen.item()
            n_batches += 1

        gen_sched.step()
        crit_sched.step()

        mean_crit = total_crit / max(1, n_batches)
        mean_gp = total_gp / max(1, n_batches)
        mean_gen = total_gen / max(1, n_batches)

        print(
            f"Epoch {epoch + 1:02d}/{N_EPOCHS} | "
            f"Critic Loss: {mean_crit:.5f} | "
            f"GP: {mean_gp:.5f} | "
            f"Generator Loss: {mean_gen:.5f} |"
        )

        if _interrupted:
            break

    torch.save(generator.model.state_dict(), GENERATOR_CHECKPOINT_PATH)
    torch.save(critic.state_dict(), CRITIC_CHECKPOINT_PATH)
    print(f"Model saved -> {GENERATOR_CHECKPOINT_PATH}")

