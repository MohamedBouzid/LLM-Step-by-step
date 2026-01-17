import torch

"""
Sampling answers:
“How do we turn scores into an actual next token?”
"""

def sample_logits(
    logits,
    temperature=1.0,
    top_k=None,
    top_p=None
):
    if temperature == 0:
        return torch.argmax(logits, dim=-1)

    logits = logits / temperature

    if top_k is not None:
        logits = top_k_logits(logits, top_k)

    if top_p is not None:
        logits = top_p_logits(logits, top_p)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

#TOP-P (NUCLEUS SAMPLING)
def top_p_logits(logits, p):
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum_probs = torch.cumsum(probs, dim=-1)

    # Keep only the first token that crosses p and remove the ones after
    mask = cum_probs > p
    mask[..., 1:] = mask[..., :-1].clone()
    # This is done for edge case where all probabilities are > p
    mask[..., 0] = False

    # Remove unwanted tokens
    sorted_logits[mask] = float('-inf')
    # Restore original token order
    return logits.scatter(-1, sorted_idx, sorted_logits)

#TOP-K
def top_k_logits(logits, k):
    values, _ = torch.topk(logits, k)
    min_val = values[..., -1, None]
    return torch.where(logits < min_val, float('-inf'), logits)

def generate(model, context, max_new_tokens, **kwargs):
    for _ in range(max_new_tokens):
        logits = model(context)
        next_logits = logits[:, -1, :]
        next_token = sample_logits(next_logits, **kwargs)
        context = torch.cat([context, next_token], dim=1)
    return context

