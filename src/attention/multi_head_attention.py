import torch.nn as nn
import torch

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        assert emb_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        self.W_Q = nn.Linear(emb_dim, emb_dim)
        self.W_K = nn.Linear(emb_dim, emb_dim)
        self.W_V = nn.Linear(emb_dim, emb_dim)
        self.W_O = nn.Linear(emb_dim, emb_dim)

    def forward(self, X):
        B, T, D = X.shape

        Q = self.W_Q(X).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(X).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(X).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)
        weights = torch.softmax(scores, dim=-1)

        out = weights @ V
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        return self.W_O(out)
