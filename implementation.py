import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 5000):
        super().__init__()

        # Create a matrix of [max_len, d_model] with positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # Apply sine to even indices, cosine to odd indices
        pe[:, 0::2] = sin(position * div_term)
        pe[:, 1::2] = cos(position * div_term)
        pe = pe.unsqueeze

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Return:
            Tensor of same shape with positional encodings added
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x