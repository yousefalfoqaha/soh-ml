import torch
from torch import nn


class ConditionalEncoding(nn.Module):
    def __init__(self, n_conditions: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Linear(n_conditions, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence, conditions):

        # (batch_size, 1, embed_dim)
        bias = self.embedding(conditions).unsqueeze(1)

        # (batch_size, window_length, embed_dim)
        return self.dropout(sequence + bias)


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
        embeddings = embeddings + attention

        pre_feedforward_embeddings = self.post_attention_norm(embeddings)
        embeddings = embeddings + self.feedforward(pre_feedforward_embeddings)

        return embeddings


class SohEstimator(nn.Module):
    def __init__(
        self,
        input_features: int,
        n_conditions: int,
        embedding_dim: int,
        n_heads: int,
        n_blocks: int,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
        max_length: int = 1000,
    ):
        super().__init__()

        self.input_signal_embedding = nn.Linear(input_features, embedding_dim)

        self.pos_encoding = PositionalEncoding(max_length, embedding_dim, dropout)
        self.conditional_encoding = ConditionalEncoding(
            n_conditions, embedding_dim, dropout
        )

        self.blocks = nn.ModuleList(
            [
                EncoderBlock(embedding_dim, n_heads, feedforward_dim, dropout)
                for _ in range(n_blocks)
            ]
        )

        self.output = nn.Linear(embedding_dim, 1)

    # X: (batch_size, max_length, X_features)
    def forward(self, X, conditions):

        # (batch_size, sequence_length, embedding_dim)
        contextual_embeddings = self.input_signal_embedding(X)
        contextual_embeddings = self.conditional_encoding(
            contextual_embeddings, conditions
        )
        contextual_embeddings = self.pos_encoding(contextual_embeddings)

        for block in self.blocks:
            contextual_embeddings = block(contextual_embeddings)

        mean_pool = torch.mean(contextual_embeddings, dim=1)

        # (batch_size, 1)
        return self.output(mean_pool)
