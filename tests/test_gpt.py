# tests/test_byte_pair_encoder.py
import pytest
from torch import Tensor
import torch
from mini_gpt.gpt import MiniGPT
import torch.nn.functional as F

def test_gpt():
    # hyperparameters
    vocab_size = 100
    max_len = 32
    emb_dim = 64
    num_heads = 4
    ff_dim = 256
    num_layers = 200

    model = MiniGPT(
        vocab_size,
        max_len,
        emb_dim,
        num_heads,
        ff_dim,
        num_layers
    )

    # dummy batch
    idx = torch.randint(0, vocab_size, (2, 10))
    targets = torch.randint(0, vocab_size, (2, 10))

    logits = model(idx)
    loss = compute_loss(logits, targets)

    print("logits shape:", logits.shape, True)  # (2, 10, vocab_size)
    assert torch.Size([2, 10, 100]) == logits.shape
    print("loss:", loss.item(), True)
    assert 5 > loss.item() 

def compute_loss(logits, targets):
    """
    logits:  (B, T, vocab_size)
    targets: (B, T)
    """
    B, T, V = logits.shape
    logits = logits.view(B * T, V)
    targets = targets.view(B * T)
    return F.cross_entropy(logits, targets)
