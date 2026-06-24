from torch import nn


class SohConditioning(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.soh_embedding = nn.Linear(1, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence, soh):
        # (batch_size, 1, embed_dim)
        bias = self.soh_embedding(soh).unsqueeze(1)

        # (batch_size, window_length, embed_dim)
        return self.dropout(sequence + bias)
