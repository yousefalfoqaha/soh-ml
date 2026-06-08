from torch import nn


class LstmModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        lstm_outputs = self.lstm(X)
        return self.output(lstm_outputs)
