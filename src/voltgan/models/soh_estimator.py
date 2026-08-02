from pathlib import Path

import torch
import torch.nn as nn

from voltgan.config import (
    DROPOUT,
    ESTIMATOR_BASE_CHANNELS,
    ESTIMATOR_GRU_HIDDEN_SIZE,
    ESTIMATOR_GRU_N_LAYERS,
    ESTIMATOR_INPUT_FEATURES,
    ESTIMATOR_KERNEL_SIZE,
    ESTIMATOR_N_CONDITIONS,
    ESTIMATOR_STRIDE,
    LEAKY_SLOPE,
)


class SohEstimator(nn.Module):
    def __init__(
        self,
        input_features: int,
        n_conditions: int,
        base_channels: int,
        stride: int,
        gru_hidden_size: int,
        gru_n_layers: int,
        dropout: float = 0.1,
        kernel_size: int = 7,
    ):
        super().__init__()
        self.n_conditions = n_conditions
        padding = (kernel_size - stride) // 2

        self.conv_stack = nn.Sequential(
            nn.Conv1d(
                input_features,
                base_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
            ),
            nn.LeakyReLU(LEAKY_SLOPE),
            self._block(
                base_channels, 2 * base_channels, kernel_size, stride, padding, dropout
            ),
            self._block(
                2 * base_channels,
                4 * base_channels,
                kernel_size,
                stride,
                padding,
                dropout,
            ),
        )

        conv_out_channels = 4 * base_channels

        self.cond_scale = nn.Linear(n_conditions, conv_out_channels)
        self.cond_shift = nn.Linear(n_conditions, conv_out_channels)
        self.film_dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(
            input_size=conv_out_channels,
            hidden_size=gru_hidden_size,
            num_layers=gru_n_layers,
            batch_first=True,
            bidirectional=True,
        )

        gru_out_dim = gru_hidden_size * 2

        self.attn = nn.Sequential(
            nn.Linear(gru_out_dim, gru_hidden_size),
            nn.Tanh(),
            nn.Linear(gru_hidden_size, 1),
        )

        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(gru_out_dim, 1)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding, dropout):
        channels_per_group = 16
        num_groups = out_channels // channels_per_group

        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.GroupNorm(num_groups, out_channels),
            nn.LeakyReLU(LEAKY_SLOPE),
            nn.Dropout(dropout),
        )

    def forward(self, X, conditions):
        x = X.permute(0, 2, 1)
        x = self.conv_stack(x)
        x = x.permute(0, 2, 1)

        scale = self.cond_scale(conditions).unsqueeze(1)
        shift = self.cond_shift(conditions).unsqueeze(1)
        x = self.film_dropout(x * (1.0 + scale) + shift)

        gru_out, _ = self.gru(x)

        attn_scores = self.attn(gru_out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = (gru_out * attn_weights).sum(dim=1)
        pooled = self.dropout(pooled)

        return self.output(pooled)


class SohEstimatorClient:
    def __init__(
        self,
        device: str,
        checkpoint_path: Path | None = None,
        is_training: bool = False,
    ):
        self.device = device

        self.model = SohEstimator(
            input_features=ESTIMATOR_INPUT_FEATURES,
            n_conditions=ESTIMATOR_N_CONDITIONS,
            base_channels=ESTIMATOR_BASE_CHANNELS,
            stride=ESTIMATOR_STRIDE,
            kernel_size=ESTIMATOR_KERNEL_SIZE,
            gru_hidden_size=ESTIMATOR_GRU_HIDDEN_SIZE,
            gru_n_layers=ESTIMATOR_GRU_N_LAYERS,
            dropout=DROPOUT if is_training else 0.0,
        ).to(self.device)

        if checkpoint_path and checkpoint_path.exists():
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            )

        if is_training:
            self.model.train()
        else:
            self.model.eval()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def finetune(self):
        self.train()

        for param in self.model.parameters():
            param.requires_grad_(False)

        for param in self.model.attn.parameters():
            param.requires_grad_(True)

        for param in self.model.output.parameters():
            param.requires_grad_(True)

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def __call__(self, X: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        return self.model(X, conditions)
