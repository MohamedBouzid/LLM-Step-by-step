import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from bpe.model.dataset import NextTokenDataset
from bpe.model.model import TinyLM
import torch.nn as nn

# Hyperparameters
VOCAB_SIZE = 50
EMB_DIM = 32
LR = 1e-3
EPOCHS = 300

class Trainer:
    
    loss_fn = nn.CrossEntropyLoss()
    dataset: NextTokenDataset
    loader: DataLoader
    model: TinyLM

    def __init__(self, token_sequence):
        self.dataset = NextTokenDataset(token_sequence)
        self.loader = DataLoader(self.dataset, batch_size=16, shuffle=True)
        self.model = TinyLM(VOCAB_SIZE, EMB_DIM)
    
    def run(self):
        optimizer = optim.Adam(self.model.parameters(), lr=LR)

        for epoch in range(EPOCHS):
            total_loss = 0

            for X, y in self.loader:
                optimizer.zero_grad()
                logits = self.model(X)
                loss = self.loss_fn(logits, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch}: loss = {total_loss:.4f}")


    def predict_next(self, context):
        context = self.pad_inputs([context])
        x = torch.tensor(context)
        logits = self.model(x)
        return logits.argmax(dim=-1).item()
    
    CONTEXT_LEN = 5
    PAD_ID = 0
    
    def pad_inputs(self, inputs):
        padded = []
        for seq in inputs:
            if len(seq) < self.CONTEXT_LEN:
                seq = seq + [self.PAD_ID] * (self.CONTEXT_LEN - len(seq))
            else:
                seq = seq[-self.CONTEXT_LEN:]
            padded.append(seq)

        return padded
