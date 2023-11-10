import torch
import math
from torch import nn


class SinCosPositionalEncoding(nn.Module):
    def __init__(self, num_tokens, d_model, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(num_tokens).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(num_tokens, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x, positions=None):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        if positions is None:
            x = x + self.pe
        else:
            x = x + self.pe[positions, :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, num_tokens, d_model, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.num_tokens = num_tokens
        self.embedding = torch.nn.Embedding(
            num_embeddings=num_tokens, embedding_dim=d_model
        )

    def forward(self, x, positions=None):
        if positions is None:
            positions = torch.arange(self.num_tokens, device=x.device)
        else:
            positions = positions.int()
        x = x + self.embedding(positions)
        return self.dropout(x)


class Mean(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
        
    def forward(self, x):
        return torch.mean(x, dim=self.dim)
    

class ReplicateInputAttentionWrapper(nn.Module):
    def __init__(self, wrapped, n=1):
        super().__init__()
        self.n = n
        self.wrapped = wrapped
        
    def forward(self, x):
        replicate_x = [x] * self.n
        return self.wrapped(*replicate_x)[0]


class NoisyAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**-0.5
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        
    def forward(self, x, value_noise=None):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        if value_noise is not None:
            v += value_noise
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        return out, attn