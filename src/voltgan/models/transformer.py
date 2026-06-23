import torch
from torch import nn

from voltgan.models import PositionalEncoding, SohConditioning


class BatteryEncoderTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        window_length: int,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.signal_embedding = nn.Linear(2, embed_dim)

        self.pos_encoding = PositionalEncoding(window_length + 1, embed_dim, dropout)
        self.soh_conditioning = SohConditioning(embed_dim, dropout)

        self.self_attention = nn.MultiheadAttention(
            embed_dim, n_heads, dropout, batch_first=True
        )

        self.attention_norm = nn.LayerNorm(embed_dim)

        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embed_dim),
        )

        self.ffn_norm = nn.LayerNorm(embed_dim)

        self.voltage_head = nn.Linear(embed_dim, 1)
        self.temperature_head = nn.Linear(embed_dim, 1)

    def forward(self, X, initial_conditions):
        # (B, 2)
        initial_voltage_temp = initial_conditions[:, 0:2]
        # (B, 2)
        soh = initial_conditions[:, 2:3]

        # (B, 1, embed_dim)
        boundary_token = self.signal_embedding(initial_voltage_temp).unsqueeze(1)
        # (B, window_length, embed_dim)
        sequence_embeddings = self.signal_embedding(X)

        # (B, window_length + 1, embed_dim)
        combined_sequence = torch.cat([boundary_token, sequence_embeddings], dim=1)

        contextual_sequence = self.pos_encoding(combined_sequence)
        contextual_sequence = self.soh_conditioning(contextual_sequence, soh)

        attention, _ = self.self_attention(
            query=contextual_sequence,
            key=contextual_sequence,
            value=contextual_sequence,
        )
        norm_sequence = self.attention_norm(contextual_sequence + attention)

        transformer_output = self.ffn_norm(
            norm_sequence + self.feedforward(norm_sequence)
        )

        # (B, window_length, embed_dim)
        target_sequence = transformer_output[:, 1:]

        # (B, window_length, 1)
        predicted_voltage = self.voltage_head(target_sequence)
        # (B, window_length, 1)
        predicted_temperature = self.temperature_head(target_sequence)

        # (B, window_length, 2)
        return torch.cat([predicted_voltage, predicted_temperature], dim=2)
