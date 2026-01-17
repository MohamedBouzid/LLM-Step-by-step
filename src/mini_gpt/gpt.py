import torch.nn as nn
import torch
from transformer.transformer_block import TransformerBlock 

class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len,
        emb_dim,
        num_heads,
        ff_dim,
        num_layers
    ):
        super().__init__()

        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(max_len, emb_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(emb_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocab_size)

    def forward(self, idx):
        """
        idx: (B, T) token ids
        """
        B, T = idx.shape

        positions = torch.arange(T, device=idx.device).unsqueeze(0)
        X = self.tok_emb(idx) + self.pos_emb(positions)

        for block in self.blocks:
            X = block(X)

        X = self.ln_f(X)
        logits = self.lm_head(X)

        return logits
