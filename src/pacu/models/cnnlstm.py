import torch.nn as nn
import torch
import pacu.models.utils as utils

class CNNLSTM(nn.Module):

    default = {
        "out_channels":32,
        "kernel_size":3,
        "padding":1,
        "layers":(32,64)
    }

    def __init__(self, input_dim: int, options: dict):
        super().__init__()
        
        if not options:
            options = self.default
        else:
            if not utils.is_valid_cnn(self.default, options):
                print("Invalid options passed to CNNLSTM")
                print(f"CNNLSTM options: {self.default.keys()}")
                exit(1)

        self.conv = nn.Conv1d(1,
            options["out_channels"],
            kernel_size=options["kernel_size"],
            padding=options["padding"]
        )

        self.lstm = nn.LSTM(options["layers"][0], options["layers"][1], batch_first=True)

        layers = utils.build_layers_cnn(options["layers"][1:])
        layers.pop()
        self.fc = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, D]
        x = torch.relu(self.conv(x)).permute(0, 2, 1)  # [B, D, F]
        _, (h_n, _) = self.lstm(x)
        return self.sigmoid(self.fc(h_n[-1])).squeeze()


