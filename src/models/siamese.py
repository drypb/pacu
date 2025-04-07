import torch.nn as nn
import torch

class SiameseNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward_once(self, x):
        return self.embedding(x)

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        distance = torch.abs(out1 - out2)
        return self.fc(distance).squeeze()

