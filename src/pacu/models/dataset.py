
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class SiameseDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X) // 2

    def __getitem__(self, idx):
        i1 = idx * 2
        i2 = i1 + 1
        return self.X[i1], self.X[i2], torch.tensor(float(self.y[i1] == self.y[i2]))


