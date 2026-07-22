import torch
import torch.nn as nn


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
            nn.GELU(),
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

        self.dropout = nn.Dropout(dropout)

        self.output = nn.Linear(gru_hidden_size * 2, 1)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding, dropout):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, X, conditions):
        x = X.permute(0, 2, 1)
        x = self.conv_stack(x)
        x = x.permute(0, 2, 1)

        scale = self.cond_scale(conditions).unsqueeze(1)
        shift = self.cond_shift(conditions).unsqueeze(1)
        x = self.film_dropout(x * (1.0 + scale) + shift)

        _, h_n = self.gru(x)

        pooled = torch.cat([h_n[-2], h_n[-1]], dim=1)
        pooled = self.dropout(pooled)

        return self.output(pooled)
