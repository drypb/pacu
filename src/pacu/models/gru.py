import torch.nn as nn
import pacu.models.utils as utils

class TorchGRU(nn.Module):

    default = {
        "hidden_dim" : 64
    }

    def __init__(self, input_dim: int, options: dict):
        super().__init__()

        if not options:
            options = self.default
        else:
            if not utils.is_valid(self.default, options):
                print("Invalid options for GRU")
                print(f"GRU options: {self.default.keys()}")
                exit(1)


        self.gru = nn.GRU(input_dim, options["hidden_dim"], batch_first=True)
        self.fc = nn.Linear(options["hidden_dim"], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, 10, 1)
        _, h_n = self.gru(x)
        out = self.fc(h_n[-1])
        return self.sigmoid(out).squeeze()


