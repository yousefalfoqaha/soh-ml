import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, max_sequence_length: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.time_step_embeddings = nn.Parameter(
            torch.randn(max_sequence_length, embed_dim) * 0.02
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence):
        sequence_length = sequence.size(1)

        # (sequence_length, embed_dim)
        bias = self.time_step_embeddings[:sequence_length]

        # (sequence_length, embed_dim)
        return self.dropout(sequence + bias)
