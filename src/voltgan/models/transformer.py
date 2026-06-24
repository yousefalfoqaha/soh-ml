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
        self.pre_attention_norm = nn.LayerNorm(embedding_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embedding_dim),
        )
        self.post_attention_norm = nn.LayerNorm(embedding_dim)

    def forward(self, embeddings):
        pre_attention_embeddings = self.pre_attention_norm(embeddings)
        attention, _ = self.self_attention(
            query=pre_attention_embeddings,
            key=pre_attention_embeddings,
            value=pre_attention_embeddings,
            need_weights=False,
        )
        pre_feedforward_embeddings = self.post_attention_norm(embeddings + attention)

        return embeddings + self.feedforward(pre_feedforward_embeddings)


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
        boundary_embedding = self.initial_state_embedding(initial_state).unsqueeze(1)

        # (batch_size, window_length, embedding_dim)
        input_signal_embeddings = self.input_signal_embedding(X)

        # (batch_size, window_length + 1, embedding_dim)
        combined_embeddings = torch.cat(
            [boundary_embedding, input_signal_embeddings], dim=1
        )
        contextual_embeddings = self.pos_encoding(combined_embeddings)
        contextual_embeddings = self.soh_conditioning(contextual_embeddings, soh)

        for block in self.blocks:
            contextual_embeddings = block(contextual_embeddings)

        # (batch_size, window_length, embedding_dim)
        target_embeddings = contextual_embeddings[:, 1:]

        # (batch_size, window_length, 1)
        predicted_voltage = self.voltage_head(target_embeddings)
        predicted_temperature = self.temperature_head(target_embeddings)

        # (batch_size, window_length, 2)
        return torch.cat([predicted_voltage, predicted_temperature], dim=2)
