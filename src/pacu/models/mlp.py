import torch.nn as nn
import models.utils as utils

class TorchMLP(nn.Module):
    default = {
        "layers": (18,)#baseado no numero de features
    }

    def __init__(self, input_dim: int, options: dict):
        super().__init__()

        if not options:
            options = self.default
        else:
            if not utils.is_valid(self.default, options):
                print("Invalid options for MLP") 
                print(f"MLP options: {self.default.keys()}") 
                exit(1)

        layers = utils.build_layers(input_dim, options["layers"])
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze()

