import torch
from torch.utils.data import Dataset

class NextTokenDataset(Dataset):

    CONTEXT_LEN = 5
    PAD_ID = 0
    
    def __init__(self, token_sequence):
        inputs, targets = self.build_samples(token_sequence)
        padded_inputs = self.pad_inputs(inputs)
        self.X = torch.tensor(padded_inputs, dtype=torch.long)
        self.y = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def build_samples(self, token_sequences):
        inputs = []
        targets = []

        for seq in token_sequences:
            for i in range(1, len(seq)):
                inputs.append(seq[:i])
                targets.append(seq[i])

        return inputs, targets

    def pad_inputs(self, inputs):
        padded = []
        for seq in inputs:
            if len(seq) < self.CONTEXT_LEN:
                seq = seq + [self.PAD_ID] * (self.CONTEXT_LEN - len(seq))
            else:
                seq = seq[-self.CONTEXT_LEN:]
            padded.append(seq)

        return padded
