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
        kernel_size: int = 7,
    ):
        super().__init__()
        self.n_conditions = n_conditions
        self.noise_dim = noise_dim
        stride = 5

        padding = (kernel_size - stride) // 2
        in_channels = input_features + n_conditions + noise_dim

        self.input_feature_encoder = nn.Sequential(
            nn.Conv1d(
                in_channels,
                base_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
            ),
            nn.LeakyReLU(LEAKY_SLOPE),
            self._encoder_block(
                base_channels, 2 * base_channels, kernel_size, stride, padding
            ),
            self._encoder_block(
                2 * base_channels, 4 * base_channels, kernel_size, stride, padding
            ),
            self._encoder_block(
                4 * base_channels, 8 * base_channels, kernel_size, stride, padding
            ),
        )

        self.gru = nn.GRU(
            input_size=8 * base_channels,
            hidden_size=latent_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        self.voltage_decoder = nn.Sequential(
            nn.Conv1d(
                2 * latent_size,
                8 * base_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                bias=False,
            ),
            nn.ReLU(),
            self._decoder_block(
                8 * base_channels,
                4 * base_channels,
                kernel_size,
                stride,
                padding,
                1,
            ),
            self._decoder_block(
                4 * base_channels,
                2 * base_channels,
                kernel_size,
                stride,
                padding,
                1,
            ),
            self._decoder_block(
                2 * base_channels,
                base_channels,
                kernel_size,
                stride,
                padding,
                1,
            ),
        )

        self.temperature_decoder = nn.Sequential(
            nn.Conv1d(
                (2 * latent_size),
                8 * base_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                bias=False,
            ),
            nn.ReLU(),
            self._decoder_block(
                8 * base_channels,
                4 * base_channels,
                kernel_size,
                stride,
                padding,
                1,
            ),
            self._decoder_block(
                4 * base_channels,
                2 * base_channels,
                kernel_size,
                stride,
                padding,
                1,
            ),
            self._decoder_block(
                2 * base_channels,
                base_channels,
                kernel_size,
                stride,
                padding,
                1,
            ),
        )

        self.voltage_output = nn.Conv1d(base_channels, 1, kernel_size=1)
        self.temperature_output = nn.Conv1d(2 * base_channels, 1, kernel_size=1)

    def _encoder_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.InstanceNorm1d(out_channels, affine=True),
            nn.LeakyReLU(LEAKY_SLOPE),
        )

    def _decoder_block(
        self, in_channels, out_channels, kernel_size, stride, padding, output_padding
    ):
        return nn.Sequential(
            nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                output_padding,
                bias=False,
            ),
            # nn.InstanceNorm1d(out_channels, affine=True),
            nn.LeakyReLU(LEAKY_SLOPE),
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
