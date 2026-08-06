from torch import nn


class SomeBullshit(nn.Module):
    def __init__(self):
        self.lstm = nn.LSTM(input_size=64, hidden_size=128)

    def forward(self, X):
        return self.lstm(X)
