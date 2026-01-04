import pytest
import torch
from attention.multi_head_attention import MultiHeadAttention
from torch import Tensor 

@pytest.fixture
def attention():
    return MultiHeadAttention(4, 2)

def test_multi_head_attention_output(attention):
    X =Tensor([
        [
            [0.20, 0.10, 0.40, 0.30],   # "I"
            [0.90, 0.50, 0.10, 0.20],   # "love"
            [0.30, 0.80, 0.60, 0.10]    # "pizza"
        ]
        ])
    y=attention(X)
    assert y.shape == torch.Size([1, 3, 4])