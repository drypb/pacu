
from cuml.ensemble import RandomForestClassifier
from cuml.linear_model import LogisticRegression
from cuml.svm import SVC
from cuml.naive_bayes import GaussianNB
from cuml.neighbors import KNeighborsClassifier
from cuml.model_selection import train_test_split
from cuml.preprocessing import MinMaxScaler

from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report, accuracy_score  
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier  
from lightgbm import LGBMClassifier


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

_MODELS_TORCH = {
    "mlp"     : lambda n,o: pacu.models.mlp.TorchMLP(n,o),
    "charcnn" : lambda n,o: pacu.models.charcnn.CharCNN(n,o),
    "cnnlstm" : lambda n,o: pacu.models.cnnlstm.CNNLSTM(n,o),
    "gru"     : lambda n,o: pacu.models.gru.TorchGRU(n,o),
    "lstm"    : lambda n,o: pacu.models.lstm.TorchLSTM(n,o),
    "siamese" : lambda n,o: pacu.models.siamese.SiameseNet(n,o)
}

_MODELS_SKLEARN = {
    "rf"     : RandomForestClassifier(n_estimators=100, random_state=42),
    "logreg" : LogisticRegression(max_iter=1000, random_state=42),
    "svm"    : SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
    "gb"     : GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    "nb"     : GaussianNB(),
    "lgm"    : LGBMClassifier(random_state=42),
    "xgb"    : XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "knn"    : KNeighborsClassifier(n_neighbors=5),
    "ada"    : AdaBoostClassifier(n_estimators=50, random_state=42)
}

_MODELS = list(_MODELS_TORCH.keys()) + list(_MODELS_SKLEARN.keys())

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
    mode       : str          = field(init=False)


    def _init_torch(self) -> None:
        dev = "cpu"
        if torch.cuda.is_available():
            dev = "cuda"

        self.device = torch.device(dev)

        self.model  = _MODELS_TORCH[self.model_name](self.input_dim, self.options).to(self.device)
        self.loader = DataLoader(
            _DATASET.get(self.model_name, TabularDataset)(self.x_train, self.y_train),
            batch_size = self.batch_size,
            shuffle=True
        )

    
    def _init_sklearn(self) -> None:
        self.model = _MODELS_SKLEARN[self.model_name]


    def __post_init__(self) -> None:

        if self.model_name in _MODELS_SKLEARN:
            self.mode = "sklearn"
            self._init_sklearn()
        else:
            self.mode = "torch"
            self._init_torch()


    def _train_torch(self, epochs: int) -> None:

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


    def _train_sklearn(self) -> None:
        print(f"[{self.model_name.upper()}] Training...")
        self.model.fit(self.x_train, self.y_train)


    def train(self, epochs: int) -> None:

        if self.mode == "torch":
            self._train_torch(epochs)
        else:
            self._train_sklearn()


    def _accuracy_torch(self) -> None:

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
        print(f"[{self.model_name.upper()}] Accuracy: {acc:.4f}")


    def _acurracy_sklearn(self) -> None:
        y_pred = self.model.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"[{self.model_name.upper()}] Accuracy: {accuracy:.4f}")
        #print(classification_report(self.y_test, y_pred))


    def accuracy(self) -> None:
        if self.mode == "torch":
            self._accuracy_torch()
        else:
            self._acurracy_sklearn()
