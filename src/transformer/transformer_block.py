from attention.multi_head_attention import MultiHeadAttention
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, ff_dim):
        super().__init__()

        self.ln1 = nn.LayerNorm(emb_dim)
        self.attn = MultiHeadAttention(emb_dim, num_heads)

        self.ln2 = nn.LayerNorm(emb_dim)
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, emb_dim)
        )

    def forward(self, X):
        X = X + self.attn(self.ln1(X))
        X = X + self.ff(self.ln2(X))
        return X
