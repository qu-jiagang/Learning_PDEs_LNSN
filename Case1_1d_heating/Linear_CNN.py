import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 1, 3, 1, 0, bias=False),
            nn.ConstantPad1d(1,0)
        )

    def forward(self, x):
        x = self.layer1(x)
        return x


def loss_func(determination, x):
    loss = F.mse_loss(determination, x)
    return loss


Nx = 128        # number of grids
T  = 10         # total Time
dt = 0.01       # Time step: small dt=0.001; large dt=0.0025;
Ts = int(T/dt)  # total steps
mu = 0.01       # parameter

data = np.fromfile('dataset/dataset_1_'+str(dt)+'.dat').reshape([Ts,1,Nx])

Network = Net().cuda()
optimizer = optim.Adam(Network.parameters(), lr=1.0e-5)

data = torch.from_numpy(data).float()
train_data_input  = Variable(data[0:10].reshape((-1,1,128))).cuda()
train_data_output = Variable(data[1:11].reshape((-1,1,128))).cuda()

loss_base, epoch = 1.0, -1
while epoch < 1000000:
    epoch = epoch + 1
    x_reconst = Network(train_data_input)
    loss = loss_func(x_reconst, train_data_output)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss = loss.cpu().data.numpy()

    print(epoch, loss)

    if epoch > 0 and loss < loss_base:
        loss_base = loss
        torch.save(Network, 'CNN_'+str(dt)+'.net')

