import torch
from torch import nn


class BatteryGruModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        conditions_size: int,
        hidden_size: int,
        output_size: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.output_size = output_size
        self.conditions = conditions_size
        self.gru = nn.GRU(
            input_size + conditions_size,
            hidden_size,
            n_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, X, conditions, start_values, initial_hidden_state=None):
        # (batch_size, window_length, conditions_size)
        conditions = conditions.unsqueeze(1).expand([-1, X.size(1), -1])

        # (batch_size, window_length, input_size + conditions_size)
        X = torch.cat([X, conditions], dim=2)

        # 1: (batch_size, window_length, hidden_size)
        # 2: (n_layers, batch_size, hidden_size)
        outputs, last_hidden_state = self.gru(X, initial_hidden_state)

        # (batch_size, window_length, output_size)
        deltas = self.output(outputs)

        # (batch_size, 1, output_size)
        start_values = start_values.unsqueeze(1)

        return start_values + deltas, last_hidden_state
