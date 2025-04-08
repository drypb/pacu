import torch.nn as nn 
import torch
import pacu.models.utils as utils

class CharCNN(nn.Module):

    default = {
        "layers":(64,32),
        "kernel_size":3,
        "out_channels":64,
        "padding":1
    }

    def __init__(self, input_dim: int, options: dict):
        super().__init__()

        if not options:
            options = self.default
        else:
            if not utils.is_valid_cnn(self.default, options):
                print("Invalid options passed to CharCNN")
                print(f"CharCNN options: {self.default.keys()}")
                exit(1)

        self.conv1 = nn.Conv1d(in_channels=1,
             out_channels=options["out_channels"],
             kernel_size=options["kernel_size"],
             padding=options["padding"]
        )

        self.pool = nn.AdaptiveMaxPool1d(1)

        layers=utils.build_layers_cnn(options["layers"])
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, input_dim]
        x = self.pool(torch.relu(self.conv1(x))).squeeze(-1)
        return self.fc(x).squeeze()


