import torch
import torch.nn as nn


def _group_norm(channels: int, num_groups: int = 8) -> nn.GroupNorm:
    return nn.GroupNorm(min(num_groups, channels), channels)


class BatteryConvGenerator(nn.Module):
    def __init__(
        self,
        input_features: int,
        output_feature: int,
        n_conditions: int,
        base_channels: int,
        latent_size: int,
        kernel_size: int = 7,
    ):
        super().__init__()
        self.n_conditions = n_conditions

        padding = (kernel_size - 1) // 2

        self.encoder = nn.Sequential(
            nn.Conv1d(
                input_features + n_conditions,
                base_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            _group_norm(base_channels),
            nn.ReLU(),
            nn.Conv1d(
                base_channels,
                2 * base_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
            ),
            _group_norm(2 * base_channels),
            nn.ReLU(),
            nn.Conv1d(
                2 * base_channels,
                4 * base_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
            ),
            _group_norm(4 * base_channels),
            nn.ReLU(),
        )

        self.latent = nn.GRU(
            input_size=4 * base_channels,
            hidden_size=latent_size,
            num_layers=2,
            batch_first=True,
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                latent_size, 4 * base_channels, kernel_size=kernel_size, padding=padding
            ),
            _group_norm(4 * base_channels),
            nn.ReLU(),
            nn.ConvTranspose1d(
                4 * base_channels,
                2 * base_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                output_padding=1,
            ),
            _group_norm(2 * base_channels),
            nn.ReLU(),
            nn.ConvTranspose1d(
                2 * base_channels,
                base_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                output_padding=1,
            ),
            _group_norm(base_channels),
            nn.ReLU(),
        )

        self.out_conv = nn.Conv1d(base_channels, output_feature, kernel_size=1)

    def forward(self, X, conditions):
        _, sequence_length, _ = X.shape

        x = X.permute(0, 2, 1)
        cond_expanded = conditions.unsqueeze(-1).expand(-1, -1, sequence_length)
        x = torch.cat([x, cond_expanded], dim=1)

        encoder_output = self.encoder(x)

        latent_input = encoder_output.permute(0, 2, 1)
        latent_output, _ = self.latent(latent_input)

        decoder_input = latent_output.permute(0, 2, 1)
        decoder_output = self.decoder(decoder_input)

        output = self.out_conv(decoder_output)
        output = output.permute(0, 2, 1)

        return output[:, :sequence_length, :]
