import torch
import torch.nn as nn

from voltgan.config import LEAKY_SLOPE


class Critic(nn.Module):
    def __init__(
        self,
        input_features: int,
        n_conditions: int,
        base_channels: int,
        dropout: float,
        kernel_size: int = 7,
    ):
        super().__init__()
        self.n_conditions = n_conditions

        stride = 5
        padding = (kernel_size - stride) // 2
        in_channels = input_features + n_conditions

        self.critic = nn.Sequential(
            nn.Conv1d(
                in_channels,
                base_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
            ),
            nn.LeakyReLU(LEAKY_SLOPE),
            self._block(
                base_channels, base_channels * 2, kernel_size, 2, padding, dropout
            ),
            self._block(
                base_channels * 2, base_channels * 4, kernel_size, 2, padding, dropout
            ),
            self._block(
                base_channels * 4, base_channels * 8, kernel_size, 2, padding, dropout
            ),
            nn.Conv1d(
                base_channels * 8,
                1,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
            ),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding, dropout):
        return nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.InstanceNorm1d(out_channels, affine=True),
            nn.LeakyReLU(LEAKY_SLOPE),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        y: torch.Tensor,
        conditions: torch.Tensor,
    ):
        T = y.size(1)
        c = conditions.unsqueeze(1).expand(-1, T, -1)
        x = torch.cat([y, c], dim=-1)

        x = x.permute(0, 2, 1)

        output = self.critic(x)

        return torch.mean(output, dim=2).squeeze(1)
