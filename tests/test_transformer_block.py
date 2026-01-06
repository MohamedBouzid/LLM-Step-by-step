# tests/test_byte_pair_encoder.py
import pytest
from torch import Tensor
import torch
from transformer.transformer_block import TransformerBlock

@pytest.fixture
def tb():
    return TransformerBlock(4, 2, 2)

def test_transformer_block(tb):
    X =Tensor([
        [
            [0.20, 0.10, 0.40, 0.30],   # "I"
            [0.90, 0.50, 0.10, 0.20],   # "love"
            [0.30, 0.80, 0.60, 0.10]    # "pizza"
        ]
        ])
    y=tb(X)
    assert y.shape == torch.Size([1, 3, 4])