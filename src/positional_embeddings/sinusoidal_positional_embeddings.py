import torch
import math

class SinusoidalPositionalEmbedding:
    seq_len: int 
    emb_dim: int

    def __init__(self, seq_len, emb_dim):
        self.seq_len = seq_len
        self.emb_dim = emb_dim

    def get_positional_embedding(seq_len, emb_dim) -> torch.Tensor:
        pe = torch.zeros(seq_len, emb_dim)
        for pos in range(seq_len):
            for i in range(0, emb_dim, 2):
                angle = pos / (10000 ** (i / emb_dim))
                pe[pos, i] = math.sin(angle)
                pe[pos, i + 1] = math.cos(angle)
        return pe
