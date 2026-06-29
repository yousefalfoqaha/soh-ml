import torch
from torch import nn

from voltgan.models.condition_encoding import ConditionEncoding
from voltgan.models.positional_encoding import PositionalEncoding


class DecoderBlock(nn.Module):
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

    def forward(self, embeddings, mask):
        pre_attention_embeddings = self.pre_attention_norm(embeddings)
        attention, _ = self.self_attention(
            query=pre_attention_embeddings,
            key=pre_attention_embeddings,
            value=pre_attention_embeddings,
            attn_mask=mask,
            need_weights=False,
        )
        embeddings = embeddings + attention

        pre_feedforward_embeddings = self.post_attention_norm(embeddings)
        embeddings = embeddings + self.feedforward(pre_feedforward_embeddings)

        return embeddings


class BatteryDecoderTransformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        condition_size: int,
        embedding_dim: int,
        n_heads: int,
        n_blocks: int,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
        max_length: int = 10000,
    ):
        super().__init__()
        self.input_signal_embedding = nn.Linear(input_size + output_size, embedding_dim)

        self.pos_encoding = PositionalEncoding(max_length, embedding_dim, dropout)
        self.condition_encoding = ConditionEncoding(
            embedding_dim, condition_size, dropout
        )

        self.blocks = nn.ModuleList(
            [
                DecoderBlock(embedding_dim, n_heads, feedforward_dim, dropout)
                for _ in range(n_blocks)
            ]
        )

        self.output_head = nn.Linear(embedding_dim, output_size)

        mask = nn.Transformer.generate_square_subsequent_mask(max_length)
        self.register_buffer("causal_mask", mask, persistent=False)

    # X:          (batch_size, sequence_length, input_size)
    # y_previous: (batch_size, sequence_length, output_size)
    # conditions: (batch_size, condition_size)
    def forward(self, X, y_previous, conditions):
        sequence_length = X.size(1)

        mask = self.causal_mask[:sequence_length, :sequence_length]

        # (batch_size, sequence_length, input_size + output_size)
        input_signal = torch.cat([X, y_previous], dim=2)

        # (batch_size, sequence_length, embedding_dim)
        contextual_embeddings = self.input_signal_embedding(input_signal)
        contextual_embeddings = self.pos_encoding(contextual_embeddings)
        contextual_embeddings = self.condition_encoding(
            contextual_embeddings, conditions
        )

        for block in self.blocks:
            contextual_embeddings = block(contextual_embeddings, mask)

        # (batch_size, sequence_length, output_size)
        return self.output_head(contextual_embeddings)
