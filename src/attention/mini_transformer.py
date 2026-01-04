import torch
import torch.nn as nn

from attention.self_attention import SelfAttention

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, emb_dim, max_len):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb   = nn.Embedding(max_len, emb_dim)

        self.attn = SelfAttention(emb_dim)
        self.ln   = nn.LayerNorm(emb_dim)

        self.head = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape

        tok = self.token_emb(x)              # (B, T, C)
        pos = self.pos_emb(torch.arange(T))  # (T, C)
        pos = pos.unsqueeze(0)               # (1, T, C)
        X = tok + pos                        # (B, T, C)
        attn_out = self.attn(X)              # (B, T, C)
        X = self.ln(X + attn_out)             # (B, T, C)
        logits = self.head(X)                # (B, T, vocab_size)
        return logits
