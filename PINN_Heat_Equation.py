import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# define neural network
class PINN_Heat_Equation(nn.Module):
    def __init__(self):
        super(PINN_Heat_Equation, self).__init__()
        # one input, nine hidden layers, and one output layer
        input_dim = 2
        hidden_dim = 20
        output_dim = 1
        num_hidden_layers = 9

        # first layer (from input to hidden)
        fc1 = nn.Linear(input_dim, hidden_dim)
        nn.init.xavier_uniform_(fc1.weight)
        nn.init.zeros_(fc1.bias)
        setattr(self, 'fc1', fc1)

        # loop hidden layers
        for i in range(2, num_hidden_layers + 2):
            layer = nn.Linear(hidden_dim, hidden_dim)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            setattr(self, f'fc{i}', layer)

        # connect last layer
        fc_last = nn.Linear(hidden_dim, output_dim)
        nn.init.xavier_uniform_(fc_last.weight)
        nn.init.zeros_(fc_last.bias)
        setattr(self, f'fc{num_hidden_layers + 2}', fc_last)

    def forward(self, x_t):
        x = torch.tanh(self.fc1(x_t))

        # apply hidden layers
        for i in range(2, 10):
            layer = getattr(self, f'fc{i}')
            x = torch.tanh(layer(x))

        # apply last layer
        x = getattr(self, 'fc10')(x)

        return x

# define loss function
def heat_loss(model, x, t):
    # merge x and t as input, so x_t = [[x1, t1], [x2, t2]]
    x_t = torch.cat([x.view(-1, 1), t.view(-1, 1)], dim = 1)
    u = model(x_t)

    # calculate autograd
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]

    # pde loss
    pde_loss = ((u_t - u_xx) ** 2).mean()

    # initial condition loss
    init_condition = u - (1 + torch.sin(x))
    ic_loss = (init_condition ** 2).mean()

    # calculate total loss
    return pde_loss + ic_loss

# prepare the training data
x_train = torch.linspace(-1, 1, 10).view(-1, 1)
t_train = torch.linspace(0, 1, 10).view(-1, 1)
x_train.requires_grad = True
t_train.requires_grad = True

# create a grid for plotting
x_train, t_train = torch.meshgrid(x_train.view(-1), t_train.view(-1))
x_train = x_train.reshape(-1, 1)
t_train = t_train.reshape(-1, 1)

# initialize and add optimizer
model = PINN_Heat_Equation()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# initialize training error list and set the number of training epochs
train_errors = []
epoch_num = 10000

# training loop
for epoch in range(epoch_num):
    # zero the gradients
    optimizer.zero_grad()

    # calculate total loss
    loss = heat_loss(model, x_train, t_train)
    loss.backward() # calculate gradients
    optimizer.step() # update weights

    # store mse loss
    train_errors.append(loss.item())

    # print train loss per epoch
    if epoch % 200 == 0:
        print(f'Epoch {epoch}, Loss {loss.item()}')

# plot training error
plt.figure()
plt.plot(range(epoch_num), train_errors)
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Training error vs Epochs')
plt.show()













