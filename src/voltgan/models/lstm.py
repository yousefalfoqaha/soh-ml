import torch
from torch import nn


class LstmModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, X, init_cond):
        _, seq_len, _ = X.shape

        init_cond_expanded = init_cond.unsqueeze(1).expand(-1, seq_len, -1)

        lstm_input = torch.cat([X, init_cond_expanded], dim=-1)

        lstm_outputs, _ = self.lstm(lstm_input)
        deltas = self.output(lstm_outputs)

        base_u_t = init_cond[:, :2].unsqueeze(1)

        return base_u_t + deltas
