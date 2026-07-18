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
        kernel_size: int = 7,
    ):
        super().__init__()
        self.n_conditions = n_conditions

        padding = (kernel_size - 1) // 2

        self.noise_encoder = nn.Sequential(
            nn.Linear(noise_dim, base_channels), nn.LeakyReLU(LEAKY_SLOPE)
        )
        self.condition_encoder = nn.Sequential(
            nn.Linear(n_conditions, base_channels), nn.LeakyReLU(LEAKY_SLOPE)
        )
        self.input_feature_encoder = nn.Sequential(
            nn.Conv1d(
                input_features,
                base_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
            ),
            nn.LeakyReLU(LEAKY_SLOPE),
        )

        self.encoder = nn.Sequential(
            self._encoder_block(
                base_channels * 3, base_channels * 6, kernel_size, 2, padding
            ),
            self._encoder_block(
                base_channels * 6, base_channels * 12, kernel_size, 2, padding
            ),
        )

        self.voltage_decoder = nn.Sequential(
            self._decoder_block(
                base_channels * 12, base_channels * 6, kernel_size, 2, padding
            ),
            self._decoder_block(
                base_channels * 6, base_channels * 3, kernel_size, 2, padding
            ),
            self._decoder_block(
                base_channels * 3, base_channels, kernel_size, 2, padding
            ),
            nn.ConvTranspose1d(
                base_channels, 1, kernel_size, 2, padding, output_padding=1
            ),
        )

        self.temperature_decoder = nn.Sequential(
            self._decoder_block(
                base_channels * 12, base_channels * 6, kernel_size, 2, padding
            ),
            self._decoder_block(
                base_channels * 6, base_channels * 3, kernel_size, 2, padding
            ),
            self._decoder_block(
                base_channels * 3, base_channels, kernel_size, 2, padding
            ),
            nn.ConvTranspose1d(
                base_channels, 1, kernel_size, 2, padding, output_padding=1
            ),
        )

    def _encoder_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.InstanceNorm1d(out_channels),
            nn.LeakyReLU(LEAKY_SLOPE),
        )

    def _decoder_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose1d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.InstanceNorm1d(out_channels),
            nn.ReLU(),
        )

    # noise: batch x noise_dim
    def forward(self, X, conditions, noise):

        # batch x input_features x sequence_length
        x = X.permute(0, 2, 1)

        # batch x base_channels x sequence_length / 2
        x_encoded = self.input_feature_encoder(x)

        # batch x base_channels
        conditions_encoded = self.condition_encoder(conditions)
        noise_encoded = self.noise_encoder(noise)

        x_encoded_sequence_length = x.shape[1]
        conditions_encoded.unsqueeze(-1).expand(-1, -1, x_encoded_sequence_length)
        noise_encoded.unsqueeze(-1).expand(-1, -1, x_encoded_sequence_length)

        encoded_inputs = torch.cat(
            [x_encoded, conditions_encoded, noise_encoded], dim=1
        )
        encoder_output = self.encoder(encoded_inputs)

        voltage_output = self.voltage_decoder(encoder_output)
        temperature_output = self.temperature_decoder(encoder_output)

        output = torch.cat([voltage_output, temperature_output], dim=1)

        return output.permute(0, 2, 1)[:, :sequence_length, :]
