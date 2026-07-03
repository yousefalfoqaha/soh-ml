import torch
from torch import nn


class DiscriminatorCNN(nn.Module):
    def __init__(
        self, input_features, n_conditions, embedding_dim, n_blocks, dropout=0.1
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_features, embedding_dim)
        self.condition = nn.Linear(n_conditions, embedding_dim)

        blocks = []
        for _ in range(n_blocks):
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(embedding_dim, embedding_dim, kernel_size=5, padding=2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        self.blocks = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, y, conditions):
        # y: (B, L, C)
        x = self.input_projection(y)

        cond = self.condition(conditions).unsqueeze(1)
        x = x + cond

        x = x.transpose(1, 2)  # (B, C, L)

        x = self.blocks(x)

        x = self.pool(x).squeeze(-1)  # (B, C)

        return self.head(x)  # (B, 1)
