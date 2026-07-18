import torch
import torch.nn as nn

from voltgan.config import LEAKY_SLOPE


class Critic(nn.Module):
    def __init__(
        self,
        input_features: int,
        n_conditions: int,
        base_channels: int,
        kernel_size: int = 7,
    ):
        super().__init__()
        self.n_conditions = n_conditions

        padding = (kernel_size - 1) // 2

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

        self.critic = nn.Sequential(
            self._block(base_channels * 2, base_channels * 4, kernel_size, 2, padding),
            self._block(base_channels * 4, base_channels * 8, kernel_size, 2, padding),
            self._block(base_channels * 8, base_channels * 16, kernel_size, 2, padding),
            nn.Conv1d(
                base_channels * 16, 1, kernel_size=kernel_size, stride=2, padding=0
            ),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.InstanceNorm1d(out_channels),
            nn.LeakyReLU(LEAKY_SLOPE),
        )

    def forward(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        conditions: torch.Tensor,
    ):
        x = X.permute(0, 2, 1)
        y_t = y.permute(0, 2, 1)
        z = torch.cat([x, y_t], dim=1)

        z_encoded = self.input_feature_encoder(z)

        conditions_encoded = self.condition_encoder(conditions)

        conditions_encoded = conditions_encoded.unsqueeze(-1).expand(
            -1, -1, z_encoded.shape[2]
        )
        encoder_input = torch.cat([z_encoded, conditions_encoded], dim=1)

        return self.critic(encoder_input)
