import torch
from torch import nn

from voltgan.models.condition_encoding import ConditionEncoding
from voltgan.models.positional_encoding import PositionalEncoding


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
        )
        embeddings = embeddings + attention

        pre_feedforward_embeddings = self.post_attention_norm(embeddings)
        embeddings = embeddings + self.feedforward(pre_feedforward_embeddings)

        return embeddings


class DiscriminatorTransformer(nn.Module):
    def __init__(
        self,
        input_features: int,
        n_conditions: int,
        embedding_dim: int,
        n_heads: int,
        n_blocks: int,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
        max_length: int = 10000,
    ):
        super().__init__()

        self.input_signal_embedding = nn.Linear(input_features, embedding_dim)

        self.pos_encoding = PositionalEncoding(max_length, embedding_dim, dropout)
        self.condition_encoding = ConditionEncoding(
            embedding_dim, n_conditions, dropout
        )

        self.blocks = nn.ModuleList(
            [
                EncoderBlock(embedding_dim, n_heads, feedforward_dim, dropout)
                for _ in range(n_blocks)
            ]
        )

        self.output_scores = nn.Sequential(
            nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, 1)
        )

    # X:          (batch_size, sequence_length, X_features)
    # y:          (batch_size, sequence_length, y_features)
    # conditions: (batch_size, condition_size)
    def forward(self, X, y, conditions):

        # (batch_size, sequence_length, input_features)
        input_signal = torch.cat([X, y], dim=2)

        # (batch_size, sequence_length, embedding_dim)
        contextual_embeddings = self.input_signal_embedding(input_signal)
        contextual_embeddings = self.pos_encoding(contextual_embeddings)
        contextual_embeddings = self.condition_encoding(
            contextual_embeddings, conditions
        )

        for block in self.blocks:
            contextual_embeddings = block(contextual_embeddings)

        # (batch_size, sequence_length, 1)
        return self.output_scores(contextual_embeddings)
