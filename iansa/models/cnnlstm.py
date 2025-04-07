import torch.nn as nn

class CNNLSTM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(32, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, D]
        x = torch.relu(self.conv(x)).permute(0, 2, 1)  # [B, D, F]
        _, (h_n, _) = self.lstm(x)
        return self.sigmoid(self.fc(h_n[-1])).squeeze()


