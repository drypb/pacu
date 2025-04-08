import torch.nn as nn
import torch


def build_layers_cnn(layer_sizes):
    layers = []
    last = layer_sizes[0]

    if len(layer_sizes) > 1:
        for i in layer_sizes[1:]:
            layers.append(nn.Linear(last,i)) 
            layers.append(nn.ReLU())
            last = i

    layers.append(nn.Linear(last,1))
    layers.append(nn.Sigmoid())

    return layers;


def build_layers(input_dim, layer_sizes):
    layers = []
    last = input_dim

    for i in layer_sizes:
        layers.append(nn.Linear(last,i)) 
        layers.append(nn.ReLU())
        last = i

    layers.append(nn.Linear(last,1))
    layers.append(nn.Sigmoid())

    return layers;



def is_valid_cnn(default: dict, options: dict):
    if  set(options.keys()) != set(default.keys()):
        print("Wrong options!!")
        return False

    if options["layers"][0] != options["out_channels"]:
        print("First layer must match out_channels")
        return False

    return True;


def is_valid(default: dict(), options: dict):
    if set(options.keys()) != set(default.keys()):
        print("Wrong options!!")
        return False
    return True;

