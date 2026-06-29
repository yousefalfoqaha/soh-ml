import torch
from torch import nn


class GeneratorGru(nn.Module):
    def __init__(
        self,
        input_features: int,
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
            input_features + noise_dim + condition_dim,
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

    # X: (batch_size, sequence_length, input_features)
    # conditions: (batch_size, n_conditions)
    # noise: (batch_size, noise_dim)
    # hidden_state: (n_layers, batch_size, hidden_size)
    def forward(self, X, conditions, noise, hidden_state=None):

        # (batch_size, condition_dim)
        conditions = self.condition_embedding(conditions)

        # (batch_size, noise_dim + condition_dim)
        noise_plus_conditions = torch.cat([noise, conditions], dim=1)

        # (batch_size, sequence_length, noise_dim + condition_dim)
        noise_plus_conditions = noise_plus_conditions.unsqueeze(1).expand(
            [-1, X.size(1), -1]
        )

        # (batch_size, sequence_length, input_features + noise_dim + condition_dim)
        X = torch.cat([X, noise_plus_conditions], dim=2)

        # (batch_size, window_length, hidden_size)
        outputs, last_hidden_state = self.gru(X, hidden_state)

        # (batch_size, window_length, output_features)
        return self.output(outputs), last_hidden_state
