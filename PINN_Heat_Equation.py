import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# define neural network
class PINN_Heat_Equation(nn.Module):
    def __init__(self):
        super(PINN_Heat_Equation, self).__init__()
        # one input, two hidden, and one output layer
        self.fc1 = nn.Linear(2, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 1)

    def forward(self, x_t):
        # Extract the spatial variable x and time variable t from the input tensor x_t
        x, t = x_t[:, 0], x_t[:, 1]

        # use cat to merge x and t into a 2D matrix
        x_t = torch.cat([x.view(-1, 1), t.view(-1, 1)], dim=1)

        # activate these layers
        x = torch.tanh(self.fc1(x_t))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        #  connect the final layer
        x = self.fc4(x)

        # return output
        return x
