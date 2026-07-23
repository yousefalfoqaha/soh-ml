import torch
import torch.nn as nn

from voltgan.config import LEAKY_SLOPE


class Generator(nn.Module):
    def __init__(
        self,
        input_features: int,
        n_conditions: int,
        base_channels: int,
        noise_dim: int,
        latent_size: int,
        dropout: float,
        kernel_size: int = 7,
    ):
        super().__init__()
        self.n_conditions = n_conditions
        self.noise_dim = noise_dim

        padding = (kernel_size - 1) // 2
        in_channels = input_features + n_conditions + noise_dim

        self.input_feature_encoder = nn.Sequential(
            nn.Conv1d(
                in_channels,
                base_channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
            nn.LeakyReLU(LEAKY_SLOPE),
            self._conv_block(
                base_channels, 2 * base_channels, kernel_size, padding, dropout
            ),
            self._conv_block(
                2 * base_channels,
                4 * base_channels,
                kernel_size,
                padding,
                dropout,
            ),
            self._conv_block(
                4 * base_channels,
                8 * base_channels,
                kernel_size,
                padding,
                dropout,
            ),
        )

        self.gru = nn.GRU(
            input_size=8 * base_channels,
            hidden_size=latent_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.temperature_output = nn.Conv1d(2 * base_channels, 1, kernel_size=1)

    def _conv_block(self, in_channels, out_channels, kernel_size, padding, dropout):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding, bias=False),
            nn.InstanceNorm1d(out_channels, affine=True),
            nn.LeakyReLU(LEAKY_SLOPE),
            nn.Dropout(dropout),
        )

    def forward(self, X, conditions, noise):
        _, sequence_length, _ = X.shape

        x = X.permute(0, 2, 1)
        cond_expanded = conditions.unsqueeze(-1).expand(-1, -1, sequence_length)
        noise_expanded = noise.unsqueeze(-1).expand(-1, -1, sequence_length)
        x = torch.cat([x, cond_expanded, noise_expanded], dim=1)

        encoder_output = self.input_feature_encoder(x)

        latent_input = encoder_output.permute(0, 2, 1)
        latent_output, _ = self.gru(latent_input)

        latent_output = latent_output.permute(0, 2, 1)

        voltage_features = self.voltage_decoder(latent_output)
        temperature_features = self.temperature_decoder(latent_output)

        combined_features = torch.cat([voltage_features, temperature_features], dim=1)
        voltage_output = self.voltage_output(voltage_features)
        temperature_output = self.temperature_output(combined_features)

        output = torch.cat([voltage_output, temperature_output], dim=1)

        return output.permute(0, 2, 1)[:, :sequence_length, :]
