import math
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.W_Q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_K = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_V = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_O = nn.Linear(emb_dim, emb_dim, bias=False)

    def forward(self, X):
        
        # B : Batch size : Number of sequences processed in parallel
        # T : Sequence length : Number of tokens per sequence 
        # C : Embedding dimension : Size of each token vector
        B, T, C = X.shape
        
        # Weights matrix shapes is [emb_dim x emb_dim]
        # Here we multily weights with embeddings matrix X
        Q = self.W_Q(X) # what I want
        K = self.W_K(X) # what you offer
        V = self.W_V(X)

        scores = Q @ K.transpose(-2, -1) 
        scores = scores / math.sqrt(C)

        # torch.ones(T, T) creates a T×T matrix of ones.
        # torch.tril keeps the lower triangle and sets everything above the diagonal to 0.
        mask = torch.tril(torch.ones(T, T, device=X.device)) # Output a matrix wi

        # Wherever mask == 0, we replace the score with -inf.
        # Why -inf? Because softmax(-inf) = 0
        # So attention weight → 0 for future tokens
        scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = torch.softmax(scores, dim=-1)
        out = weights @ V

        return self.W_O(out)
