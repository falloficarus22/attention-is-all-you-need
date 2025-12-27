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
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

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

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.scale = 1.0 / math.sqrt(d_k)

    def forward(self, Q, K, V, mask = None):
        """
        Q, K, V: shape (batch, num_heads, seq_len, d_k)
        mask: shape (batch, 1, seq_len, seq_len) or (batch, seq_len, seq_len)
        """
        # Compute raw scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            # Mask out illegal connections
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(scores, dim = -1)
        output = torch.matmul(attn_weights, V) # Weighted sum of V

        return output, attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Learned projections for queries, keys and values
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Final output linear projection
        self.W_o = nn.Linear(d_model, d_model)

        # Scaled dot-product attention module
        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(self, Q, K, V, mask = None):
        """
        Q, K, V: tensors of shape (batch_size, seq_len, d_model)
        mask: optional (batch_size, seq_len, seq_len)
        Returns: 
            output: (batch_size, seq_len, d_model)
            attn_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size = Q.size(0)

        # Linear projections and reshape for multiple heads
        # After view shape (batch_size, seq_len, num_heads, d_k)
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply scaled dot-product on all heads
        if mask is not None:
            # Expand mask for attention heads: (batch_size, 1, seq_len, seq_len)
            mask.unsqueeze(1)
        
        attn_output, attn_weights = self.attention(Q, K, V, mask = mask)

        # Concatenate layers for the final layer
        # attn_output: (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(attn_output)

        return output, attn_weights
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        return self.linear2(self.relu(self.linear1(x)))
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask = None):
        # Self-attention sublayer
        attn_output, _ = self.self_attn(x, x, x, mask = mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward sublayer
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask = None, tgt_mask = None):
        # Masked self-attention (decoder)
        attn_output, _ = self.self_attn(x, x, x, mask = tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Encoder-decoder attention
        attn_output, _ = self.enc_dec_attn(x, x, x, mask = src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        # Feed Forward network
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))

        return x
    
