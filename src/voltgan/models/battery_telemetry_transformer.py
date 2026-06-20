from torch import nn

from voltgan.models import PositionalEmbedding


class BatteryTelementryTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        window_length: int,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.current_proj = nn.Linear(1, embed_dim)

        self.pos_embed = PositionalEmbedding(window_length, embed_dim, dropout)
        self.cond_proj = nn.Linear(1, embed_dim)

        self.self_attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout, batch_first=True
        )

        self.norm1 = nn.LayerNorm(embed_dim)

        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embed_dim),
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        self.voltage_head = nn.Linear(embed_dim, 1)
        self.temp_head = nn.Linear(embed_dim, 1)
