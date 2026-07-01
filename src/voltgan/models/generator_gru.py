import torch
from torch import nn


class GeneratorGru(nn.Module):
    def __init__(
        self,
        n_conditions: int,
        hidden_size: int,
        output_features: int,
        n_layers: int,
        dropout: float,
        noise_dim: int,
        condition_dim: int,
    ):
        super().__init__()

        self.gru = nn.GRU(
            noise_dim + condition_dim,
            hidden_size,
            n_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.condition_embedding = nn.Linear(n_conditions, condition_dim)

        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_features),
        )

    # conditions: (batch_size, n_conditions)
    # noise: (batch_size, sequence_length, noise_dim)
    def forward(self, conditions, noise, hidden_state=None):

        # (batch_size, sequence_length, condition_dim)
        conditions = (
            self.condition_embedding(conditions)
            .unsqueeze(1)
            .expand(-1, noise.size(1), -1)
        )

        # (batch_size, sequence_length, noise_dim + condition_dim)
        input = torch.cat([noise, conditions], dim=2)

        outputs, last_hidden_state = self.gru(input, hidden_state)

        # (batch_size, sequence_length, output_features)
        y = self.output(outputs)

        return y, last_hidden_state
