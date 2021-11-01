import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from math import *

node = 32

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(     2, node*2, 3, 1, 1, padding_mode='circular'),
            nn.GELU(),
            nn.Conv2d(node*2,      2, 1, 1, 0, padding_mode='circular', bias=False),
        )

    def forward(self, x):
        x = self.layer1(x)
        return  x


def loss_func(determination, x):
    BCE = F.mse_loss(determination, x)
    return BCE


Network = Net().cuda()

optimizer = optim.Adam(Network.parameters(), lr=1.0e-4)

u = np.fromfile('dataset/dataset_u_0.dat').reshape([301,1,128,128])
v = np.fromfile('dataset/dataset_v_0.dat').reshape([301,1,128,128])
data = np.hstack([u,v])

data = torch.from_numpy(data).float()
train_data_input  = Variable(data[0:1].reshape((-1,2,128,128))).cuda()
train_data_output = Variable(data[1:2].reshape((-1,2,128,128))).cuda()

_loss, loss_base, epoch = 1.0, 1000000.0, -1

loss = 1000
while epoch < 1000000:
    epoch = epoch + 1

    x_reconst = Network(train_data_input)
    loss = loss_func(x_reconst, train_data_output)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss = loss.cpu().data.numpy()

    print('Case 5:', epoch, loss)

    if epoch > 0 and loss < loss_base:
        loss_base = loss
        torch.save(Network, 'CNN.net')
