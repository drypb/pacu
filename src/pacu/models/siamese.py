import torch.nn as nn
import torch
import models.utils as utils

class SiameseNet(nn.Module):

    default = {
        "layers" : (128,64)
    }

    def __init__(self, input_dim: int, options: dict):
        super().__init__()

        if not options:
            options = self.default
        else:
            if not utils.is_valid(self.default,options):
               print("Invalid options passed to Siamese Net")
               print(f"Siamese Net options: {self.default.keys()}")
               exit(1) 

        embedding = []
        last = input_dim
        for layer in options["layers"]:
            embedding.append(nn.Linear(last, layer))
            embedding.append(nn.ReLU())
            last = layer
           
        self.embedding = nn.Sequential(*embedding)
         

        self.fc = nn.Sequential(
            nn.Linear(last, 1),
            nn.Sigmoid()
        )

    def forward_once(self, x):
        return self.embedding(x)

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        distance = torch.abs(out1 - out2)
        return self.fc(distance).squeeze()

