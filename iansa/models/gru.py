import torch.nn as nn


class TorchGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, 10, 1)
        _, h_n = self.gru(x)
        out = self.fc(h_n[-1])
        return self.sigmoid(out).squeeze()


