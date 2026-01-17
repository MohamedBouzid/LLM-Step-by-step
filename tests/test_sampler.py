# tests/test_sampler.py
import torch
import torch.nn.functional as F

from sampling.sampler import sample_logits

logits = torch.tensor([[3.0, 2.0, 1.0, 0.0, -1.0]])
# shape: (1, vocab_size=5)

def test_temperature_is_zero():
    token = sample_logits(logits, temperature=0)
    assert token.item() == 0
    print("✅ Temp=0 deterministic test passed")

def test_high_temperature_increases_randomness():
    counts = torch.zeros(5)

    for _ in range(1000):
        token = sample_logits(logits, temperature=2.0)
        counts[token.item()] += 1

    print("Counts:", counts)
    assert counts[0] < 900  # not always the top token
    print("✅ High temperature randomness test passed")

def test_top_k_restricts_choices():
    k = 2
    counts = torch.zeros(5)

    for _ in range(1000):
        token = sample_logits(logits, top_k=k)
        counts[token.item()] += 1

    print("Counts:", counts)
    assert counts[2:].sum() == 0
    print("✅ Top-k restriction test passed")

def test_top_p_removes_tail_tokens():
    p = 0.8
    counts = torch.zeros(5)

    for _ in range(1000):
        token = sample_logits(logits, top_p=p)
        counts[token.item()] += 1

    print("Counts:", counts)
    assert counts[-1] == 0  # lowest-prob token removed
    print("✅ Top-p nucleus test passed")

def test_never_mask_all_token():
    logits = torch.tensor([[100.0, -100.0, -100.0]])
    token = sample_logits(logits, top_p=0.01)
    assert token.item() == 0
    print("✅ Safety test passed (top token preserved)")

def test_effect_of_temperature_on_distribution():
    import matplotlib.pyplot as plt
    temps = [0.5, 1.0, 2.0]
    for T in temps:
        probs = F.softmax(logits / T, dim=-1).squeeze()
        plt.plot(probs.numpy(), label=f"T={T}")

    plt.legend()
    plt.title("Effect of Temperature on Distribution")
    plt.show()
