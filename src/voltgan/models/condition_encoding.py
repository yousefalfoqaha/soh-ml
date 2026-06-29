from torch import nn


class ConditionEncoding(nn.Module):
    def __init__(self, embed_dim: int, conditions_size: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Linear(conditions_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence, conditions):
        # (batch_size, 1, embed_dim)
        bias = self.embedding(conditions).unsqueeze(1)

        # (batch_size, window_length, embed_dim)
        return self.dropout(sequence + bias)
