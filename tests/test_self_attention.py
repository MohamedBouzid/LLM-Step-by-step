# tests/test_byte_pair_encoder.py
import pytest
import torch
from attention.self_attention import SelfAttention
from torch import Tensor 

@pytest.fixture
def attention():
    return SelfAttention(4)

def test_self_attention_output(attention):
    X =Tensor([
        [
            [0.20, 0.10, 0.40, 0.30],   # "I"
            [0.90, 0.50, 0.10, 0.20],   # "love"
            [0.30, 0.80, 0.60, 0.10]    # "pizza"
        ]
        ])
    y=attention(X)
    print("result = ", y, True)
    print("shape = ", y.shape, True)

    assert y.shape == torch.Size([1, 3, 4])