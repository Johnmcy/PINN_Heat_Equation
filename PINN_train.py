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
                ('hidden_%d' % i, nn.Linear(hidden_size, hidden_size))
            )
            layers.append(('activation_%d' % i, act()))

        # output layer
        layers.append(('output', nn.Linear(hidden_size, output_size)))

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
        self.x_inside = torch.stack(torch.meshgrid(x, t)).reshape(2, -1).T

        # spatial and time coordinates at boundary
        bc1 = torch.stack(torch.meshgrid(x[0], t)).reshape(2, -1).T  # x=-1 boundary
        bc2 = torch.stack(torch.meshgrid(x[-1], t)).reshape(2, -1).T  # x=1 boundary
        ic = torch.stack(torch.meshgrid(x, t[0])).reshape(2, -1).T  # t=0 boundary
        self.x_boundary = torch.cat([bc1, bc2])  # merge bc1 and bc2
        self.t_boundary = ic

        # u value(temperature) at boundary
        u_ic = 1.0 + torch.sin(ic[:, 0])  # at t=0 boundary apply u=1 +sin(x)
        self.u_t_boundary = u_ic
        self.u_t_boundary = self.u_t_boundary.unsqueeze(1)

        # copy data to GPU(if there is one)
        self.x_inside = self.x_inside.to(device)
        self.x_boundary = self.x_boundary.to(device)
        self.t_boundary = self.t_boundary.to(device)
        self.u_t_boundary = self.u_t_boundary.to(device)
        self.x_inside.requires_grad = True  # calculate the gradient of X_inside
        self.x_boundary.requires_grad = True  # calculate the gradient of X_boundary

        # set criterion function as MSELoss
        self.criterion = torch.nn.MSELoss()

        # initialize epoch number
        self.epoch = 1

        # set adam optimizer
        self.adam = torch.optim.Adam(self.model.parameters(), lr=0.0001)

    def heat_loss(self):
        # zero the gradients
        self.adam.zero_grad()

        # first part: initial loss
        t_pred_boundary = self.model(self.t_boundary)  # calculate prediction value at boundary
        loss_boundary_t = self.criterion(
            t_pred_boundary, self.u_t_boundary)  # calculate MSE at boundary

        # second part: inside loss
        u_inside = self.model(self.x_inside)  # calculate prediction value at inside points

        # use autograd to calculate du/dx
        du_dx = torch.autograd.grad(
            inputs=self.x_inside,
            outputs=u_inside,
            grad_outputs=torch.ones_like(u_inside),
            retain_graph=True,
            create_graph=True
        )[0]
        Du_dx = du_dx[:, 0]  # extract du/dx
        Du_dt = du_dx[:, 1]  # extract du/dt

        # use autograd to calculate du/dxx
        du_dxx = torch.autograd.grad(
            inputs=self.x_inside,
            outputs=du_dx,
            grad_outputs=torch.ones_like(du_dx),
            retain_graph=True,
            create_graph=True
        )[0][:, 0]
        loss_equation = self.criterion(
            Du_dt, 0.1 * du_dxx)  # calculate MSE of heat equation

        # third part: loss caused by Neumann boundary condition
        u_x_boundary = self.model(self.x_boundary)

        du_dx = torch.autograd.grad(
            inputs=self.x_boundary,
            outputs=u_x_boundary,
            grad_outputs=torch.ones_like(u_x_boundary),
            retain_graph=True,
            create_graph=True
        )[0]

        Du_dx = du_dx[:, 0]  # calculate loss if boundary gradient != 0
        loss_equation_x = self.criterion(Du_dx, 0 * Du_dx)

        # calculate total loss
        loss = loss_equation + loss_boundary_t + loss_equation_x

        # calculate gradients
        loss.backward()

        # print train loss per epoch
        losses.append(loss.item())
        if self.epoch % 100 == 0:
            print(f'Epoch: {self.epoch}, Loss: {loss.item()}')
        self.epoch = self.epoch + 1
        return loss

    # training function
    def train(self):
        self.model.train()

        # print train loss per epoch
        print("Training starts")
        for i in range(5000):
            self.adam.step(self.heat_loss)

if __name__ == '__main__':
    # example PINN
    pinn = PINN()

    # training starts
    losses = []  # record loss
    pinn.train()

    # save model to a file
    torch.save(pinn.model, 'model.pth')

    # plot training error
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title('Training Error vs Epochs')
    plt.show()




