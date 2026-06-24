import torch
from torch import nn

from voltgan.models import PositionalEncoding, SohConditioning


class EncoderBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            embedding_dim, n_heads, dropout, batch_first=True
        )

        self.attention_norm = nn.LayerNorm(embedding_dim)

        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embedding_dim),
        )

        self.feedforward_norm = nn.LayerNorm(embedding_dim)

    def forward(self, embeddings):
        attention, _ = self.self_attention(
            query=embeddings,
            key=embeddings,
            value=embeddings,
        )
        norm_sequence = self.attention_norm(embeddings + attention)

        return self.feedforward_norm(norm_sequence + self.feedforward(norm_sequence))


class BatteryEncoderTransformer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        n_blocks: int,
        window_length: int,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.input_signal_embedding = nn.Linear(2, embedding_dim)
        self.initial_state_embedding = nn.Linear(2, embedding_dim)

        self.pos_encoding = PositionalEncoding(
            window_length + 1, embedding_dim, dropout
        )
        self.soh_conditioning = SohConditioning(embedding_dim, dropout)

        self.blocks = nn.ModuleList(
            [
                EncoderBlock(embedding_dim, n_heads, feedforward_dim, dropout)
                for _ in range(n_blocks)
            ]
        )

        self.voltage_head = nn.Linear(embedding_dim, 1)
        self.temperature_head = nn.Linear(embedding_dim, 1)

    def forward(self, X, initial_conditions):
        # (batch_size, 2)
        initial_state = initial_conditions[:, 0:2]
        # (batch_size, 1)
        soh = initial_conditions[:, 2:3]

        # (batch_size, 1, embedding_dim)
        boundary_token = self.initial_state_embedding(initial_state).unsqueeze(1)

        # (batch_size, window_length, embedding_dim)
        sequence_embeddings = self.input_signal_embedding(X)

        # (batch_size, window_length + 1, embedding_dim)
        combined_sequence = torch.cat([boundary_token, sequence_embeddings], dim=1)
        contextual_sequence = self.pos_encoding(combined_sequence)
        contextual_sequence = self.soh_conditioning(contextual_sequence, soh)

        for block in self.blocks:
            contextual_sequence = block(contextual_sequence)

        # (batch_size, window_length, embedding_dim)
        target_sequence = contextual_sequence[:, 1:]

        # (batch_size, window_length, 1)
        predicted_voltage = self.voltage_head(target_sequence)
        predicted_temperature = self.temperature_head(target_sequence)

        # (batch_size, window_length, 2)
        return torch.cat([predicted_voltage, predicted_temperature], dim=2)
