import torch.nn as nn
import models.utils as utils

class TorchLSTM(nn.Module):

    default = {
        "hidden_dim" : 64
    }

    def __init__(self, input_dim: int, options: int):
        super().__init__()
        
        if not options:
            options = self.default
        else:
            if not utils.is_valid(self.default, options):
                print("Invalid options for LSTM")
                print(f"LSTM options: {self.default.keys()}")
                exit(1)

        self.lstm = nn.LSTM(input_dim, options["hidden_dim"], batch_first=True)
        self.fc = nn.Linear(options["hidden_dim"], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, 10, 1)
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return self.sigmoid(out).squeeze()

