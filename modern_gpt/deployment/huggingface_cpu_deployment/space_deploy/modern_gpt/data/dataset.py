import torch
from torch.utils.data import Dataset

from .tokenizer import encode


class PackedTokenDataset(Dataset):
    def __init__(self, text, context_length, stride=None, tokenizer=None):
        token_ids = encode(text, tokenizer)
        stride = stride or context_length
        self.inputs = []
        self.targets = []
        for start in range(0, len(token_ids) - context_length, stride):
            self.inputs.append(torch.tensor(
                token_ids[start:start + context_length], dtype=torch.long
            ))
            self.targets.append(torch.tensor(
                token_ids[start + 1:start + context_length + 1], dtype=torch.long
            ))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]
