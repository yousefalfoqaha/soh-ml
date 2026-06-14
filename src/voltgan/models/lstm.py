import torch
from torch import nn


class LstmModel(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers, output_size, init_condition_size
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.hidden_state_input = nn.Linear(
            init_condition_size, num_layers * hidden_size
        )

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
        batch_size = X.shape[0]

        h_state = self.hidden_state_input(init_cond)

        h_0 = h_state.view(self.num_layers, batch_size, self.hidden_size)
        c_0 = torch.zeros_like(h_0)
        lstm_outputs, _ = self.lstm(X, (h_0, c_0))

        return self.output(lstm_outputs)

