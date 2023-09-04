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
            output_size,
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
        layers.append(('output', nn.Linear(input_size, output_size)))

        # assemble those layers
        self.layers = torch.nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.layers(x)

    # achieve physics informed neural network
    class PINN:
        def __init__(self):
            # choose GPU or CPU
            device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")

            # import parameters
            self.model = Network(
                input_size=2,
                hidden_size=16,
                output_size=1,
                depth=8,

                # activation functions for input and hidden layers
                act=torch.nn.Tanh
            ).to(device)  # store the network in GPU(if there is one)

            self.s = 0.1  # spatial step
            self.t = 0.1  # time step
            x = torch.arange(-1, 1 + self.s, self.s)  # take value from [-1,1] uniformly, note as x
            t = torch.arange(0, 1 + self.t, self.t)  # take value from [0,1] uniformly, note as t

            # merge x and t
            self.X_inside = torch.stack(torch.meshgrid(x, t)).reshape(2, -1).T

            # spatial and time coordinates at boundary
            bc1 = torch.stack(torch.meshgrid(x[0], t)).reshape(2, -1).T  # x=-1 boundary
            bc2 = torch.stack(torch.meshgrid(x[-1], t)).reshape(2, -1).T  # x=1 boundary
            ic = torch.stack(torch.meshgrid(x, t[0])).reshape(2, -1).T  # t=0 boundary
            self.X_boundary = torch.cat([bc1, bc2])  # merge bc1 and bc2
            self.T_boundary = ic

            # u value(temperature) at boundary
            u_ic = 1.0 + torch.sin(ic[:, 0])  # at t=0 boundary apply u=1 +sin(x)
            self.U_T_boundary = u_ic
            self.U_T_boundary = self.U_T_boundary.unsqueeze(1)

            # copy data to GPU(if there is one)
            self.X_inside = self.X_inside.to(device)
            self.X_boundary = self.X_boundary.to(device)
            self.T_boundary = self.T_boundary.to(device)
            self.U_T_boundary = self.U_T_boundary.to(device)
            self.X_inside.requires_grad = True  # calculate the gradient of X_inside
            self.X_boundary.requires_grad = True  # calculate the gradient of X_boundary

            # set criterion function as MSELoss
            self.criterion = torch.nn.MSELoss()

            # initialize epoch number
            self.epoch = 1

            # use adam optimizer
            self.adam = torch.optim.Adam(self.model.parameters())




