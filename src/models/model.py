
from dataclass import *

import charcnn
import cnnlstm
import siamese
import lstm
import mlp
import gru


_models = {
    "mlp" : mlp.TorchMlp,
    "gru" : gru.TorchGru
}


@dataclass
class Model:

    name: str
    input_dim: int

    
    def train():
        pass
