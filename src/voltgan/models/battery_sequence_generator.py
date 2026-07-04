import torch
from torch import nn


class BatterySequenceGenerator(nn.Module):
    def __init__(
        self,
        input_features: int,
        n_conditions: int,
        hidden_size: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(input_features, hidden_size),
            nn.Dropout(dropout),
        )

        self.cond_scale = nn.Linear(n_conditions, hidden_size)
        self.cond_shift = nn.Linear(n_conditions, hidden_size)

        self.voltage_gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.voltage_output = nn.Linear(hidden_size, 1)

        self.temperature_gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.temperature_output = nn.Linear(hidden_size, 1)

    def forward(self, X, conditions, hidden_state=None):
        h = self.input_projection(X)

        scale = self.cond_scale(conditions).unsqueeze(1)
        shift = self.cond_shift(conditions).unsqueeze(1)
        h = h * (1.0 + scale) + shift

        if hidden_state is None:
            voltage_h0, temperature_h0 = None, None
        else:
            voltage_h0, temperature_h0 = hidden_state

        voltage_out, voltage_last = self.voltage_gru(h, voltage_h0)
        temperature_out, temperature_last = self.temperature_gru(h, temperature_h0)

        voltage_y = self.voltage_output(voltage_out)
        temperature_y = self.temperature_output(temperature_out)

        y = torch.cat([voltage_y, temperature_y], dim=2)

        return y, (voltage_last, temperature_last)
