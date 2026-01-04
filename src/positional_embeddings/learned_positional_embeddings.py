import torch
import torch.nn as nn
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, emb_dim):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, emb_dim)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return self.pos_emb(positions)