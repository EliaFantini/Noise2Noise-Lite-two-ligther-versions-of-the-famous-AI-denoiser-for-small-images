import torch
from torch.utils.data.dataset import Dataset


class Dataset(Dataset):
    def __init__(self, source, target):
        self.source = source.div(255)
        self.target = target.div(255)

    def __getitem__(self, i):
        return self.source[i], self.target[i]

    def __len__(self):
        return self.source.shape[0]
