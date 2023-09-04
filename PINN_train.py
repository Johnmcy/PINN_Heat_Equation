import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import matplotlib.pyplot as plt

# define neural network
class Network(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            out_putsize,
            depth,

            # activation functions for input and hidden layers
            act = torch.nn.Tanh
    ):
        super(Network, self).__init__()

        #input layer
        layers = [('input',nn.Linear(input_size, hidden_size))]
        layers.append(('activation', act()))

        # hidden layer
        for i in range(depth):
            layers.append(
                ('hidden_%d' % i, nn.Linear(input_size, hidden_size))
            )
            layers.append(('activation_%d % i', act()))

        # output layer
        layers.append(('output', nn.Linear(input_size, out_putsize)))

        # assemble those layers
        self.layers = torch.nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.layers(x)




