
from dataclasses import *
from typing import *

from torch.utils.data import Dataset, DataLoader, TensorDataset

import torch.optim as optim
import torch.nn as nn
import torch

from pacu.models.dataset import *

import pacu.models.charcnn
import pacu.models.cnnlstm
import pacu.models.gru
import pacu.models.lstm
import pacu.models.mlp
import pacu.models.model
import pacu.models.siamese

_MODELS = {
    "mlp"     : lambda n,o: pacu.models.mlp.TorchMLP(n,o),
    "charcnn" : lambda n,o: pacu.models.charcnn.CharCNN(n,o),
    "cnnlstm" : lambda n,o: pacu.models.cnnlstm.CNNLSTM(n,o),
    "gru"     : lambda n,o: pacu.models.gru.TorchGRU(n,o),
    "lstm"    : lambda n,o: pacu.models.lstm.TorchLSTM(n,o),
    "siamese" : lambda n,o: pacu.models.siamese.SiameseNet(n,o)
}

_DATASET = {
    "siamese" : SiameseDataset
}

@dataclass
class Model:

    model_name : str
    input_dim  : int
    x_train    : Any
    x_test     : Any
    y_train    : Any
    y_test     : Any
    options    : dict
    batch_size : int = 32
    device     : torch.device = field(init=False)
    model      : nn.Module    = field(init=False)

    def __post_init__(self):
        dev = "cpu"
        if torch.cuda.is_available():
            dev = "cuda"

        self.device = torch.device(dev)

        self.model  = _MODELS[self.model_name](self.input_dim, self.options).to(self.device)
        self.loader = DataLoader(
            _DATASET.get(self.model_name, TabularDataset)(self.x_train, self.y_train),
            batch_size = self.batch_size,
            shuffle=True
        )


    def train(self, epochs: int) -> None:

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            for batch in self.loader:
                optimizer.zero_grad()
                if self.model_name == "siamese":
                    x1, x2, labels = [b.to(self.device) for b in batch]
                    preds = self.model(x1, x2)
                else:
                    x, labels =  [b.to(self.device) for b in batch]
                    preds = self.model(x)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"[{self.model_name.upper()}][Epoch {epoch+1}] Loss: {total_loss:.4f}")


    def accuracy(self) -> None:

        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            dataset_class = _DATASET.get(self.model_name, TabularDataset)
            test_ds = dataset_class(self.x_test, self.y_test)
            test_loader = DataLoader(test_ds, batch_size=32)

            for batch in test_loader:
                if self.model_name == "siamese":
                    x1, x2, labels = batch
                    x1 = x1.to(self.device)
                    x2 = x2.to(self.device)
                    labels = labels.to(self.device)
                    preds = self.model(x1, x2)
                else:
                    x, labels = batch
                    x = x.to(self.device)
                    labels = labels.to(self.device)
                    preds = self.model(x)

                predicted = (preds > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        print(f"[{self.model_name.upper()}] Acurácia: {acc:.4f}")

