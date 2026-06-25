import torch
from torch import nn


class LstmModel(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, output_size):
        super().__init__()

        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
        )

        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, X, initial_conditions):
        _, window_length, _ = X.shape

        initial_conditions_expanded = initial_conditions.unsqueeze(1).expand(
            -1, window_length, -1
        )

        lstm_input = torch.cat([X, initial_conditions_expanded], dim=-1)

        lstm_outputs, _ = self.lstm(lstm_input)
        deltas = self.output(lstm_outputs)

        base_voltage_temperature = initial_conditions[:, :2].unsqueeze(1)

        return base_voltage_temperature + deltas
