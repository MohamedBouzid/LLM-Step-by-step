import torch.nn as nn

class TinyLM(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.fc = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        # x shape: (batch, context_len)
        emb = self.embedding(x)          # (batch, context, emb_dim)
        pooled = emb.mean(dim=1)         # (batch, emb_dim)
        logits = self.fc(pooled)         # (batch, vocab_size)
        return logits
