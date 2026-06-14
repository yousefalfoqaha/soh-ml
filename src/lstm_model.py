from torch import nn


class LstmModel(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers, output_size, init_condition_size
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.h0_linear = nn.Linear(init_condition_size, num_layers * hidden_size)
        self.c0_linear = nn.Linear(init_condition_size, num_layers * hidden_size)

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

        h_state = self.h0_linear(init_cond)
        c_state = self.c0_linear(init_cond)

        h_0 = (
            h_state.view(batch_size, self.num_layers, self.hidden_size)
            .permute(1, 0, 2)
            .contiguous()
        )
        c_0 = (
            c_state.view(batch_size, self.num_layers, self.hidden_size)
            .permute(1, 0, 2)
            .contiguous()
        )

        lstm_outputs, _ = self.lstm(X, (h_0, c_0))

        return self.output(lstm_outputs)
