import torch
from torch import nn


class GeneratorGru(nn.Module):
    def __init__(
        self,
        input_features: int,
        n_conditions: int,
        hidden_size: int,
        output_features: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()

        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            n_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.input_embedding = nn.Linear(input_features + n_conditions, hidden_size)

        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_features),
        )

    # X: (batch_size, sequence_length, input_features)
    # conditions: (batch_size, n_conditions)
    def forward(self, X, conditions, hidden_state=None):

        # Expand conditions to match the sequence length
        # (batch_size, sequence_length, n_conditions)
        expanded_conditions = conditions.unsqueeze(1).expand(-1, X.size(1), -1)

        # Concatenate inputs and conditions along the feature dimension
        # (batch_size, sequence_length, input_features + n_conditions)
        combined_input = torch.cat([X, expanded_conditions], dim=2)

        # Map to hidden size
        gru_input = self.input_embedding(combined_input)

        # Pass through GRU
        outputs, last_hidden_state = self.gru(gru_input, hidden_state)

        # Project to output features
        # (batch_size, sequence_length, output_features)
        y = self.output(outputs)

        return y, last_hidden_state
