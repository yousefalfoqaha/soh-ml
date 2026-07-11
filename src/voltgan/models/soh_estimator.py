import torch
from torch import nn


class FilmConditioning(nn.Module):
    def __init__(self, n_conditions: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.cond_scale = nn.Linear(n_conditions, embed_dim)
        self.cond_shift = nn.Linear(n_conditions, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence, conditions):
        if sequence.dim() == 2:
            scale = self.cond_scale(conditions).unsqueeze(1)
            shift = self.cond_shift(conditions).unsqueeze(1)
        else:
            scale = self.cond_scale(conditions).unsqueeze(1)
            shift = self.cond_shift(conditions).unsqueeze(1)

        return self.dropout(sequence * (1.0 + scale) + shift)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class SohEstimator(nn.Module):
    def __init__(
        self,
        input_features: int,
        n_conditions: int,
        conv_channels: list[int],
        conv_kernel_sizes: list[int],
        conv_strides: list[int],
        gru_hidden_size: int,
        gru_n_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        conv_layers: list[nn.Module] = []
        in_channels = input_features
        for out_channels, kernel, stride in zip(
            conv_channels, conv_kernel_sizes, conv_strides
        ):
            conv_layers.append(ConvBlock(in_channels, out_channels, kernel, stride))
            in_channels = out_channels
        self.conv_stack = nn.Sequential(*conv_layers)

        conv_out_channels = conv_channels[-1]

        self.final_film = FilmConditioning(n_conditions, conv_out_channels, dropout)

        self.gru = nn.GRU(
            input_size=conv_out_channels,
            hidden_size=gru_hidden_size,
            num_layers=gru_n_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.output = nn.Linear(gru_hidden_size * 2, 1)

    def forward(self, X, conditions):
        x = X.permute(0, 2, 1)
        x = self.conv_stack(x)
        x = x.permute(0, 2, 1)

        x = self.final_film(x, conditions)

        _, h_n = self.gru(x)

        pooled = torch.cat([h_n[-2], h_n[-1]], dim=1)
        pooled = self.dropout(pooled)

        return self.output(pooled)
