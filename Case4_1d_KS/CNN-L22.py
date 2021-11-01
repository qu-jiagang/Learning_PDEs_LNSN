import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from math import *


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(  1, 256, 5, 1, 2, padding_mode='circular'),
            nn.GELU(),
            nn.Conv1d(256, 256, 5, 1, 2, padding_mode='circular'),
            nn.GELU(),
            nn.Conv1d(256, 256, 5, 1, 2, padding_mode='circular'),
            nn.GELU(),
            nn.Conv1d(256,   1, 1, 1, 0, bias=False),
        )

    def forward(self, x):
        x = self.layer1(x)
        return x


def loss_func(determination, x):
    loss = F.mse_loss(determination, x)
    return loss


NX = 64
# Network = Net().cuda()
Network = torch.load('CNN-L22.net')

optimizer = optim.Adam(Network.parameters(), lr=1.0e-5)

data = np.fromfile('1d_ks_L22.dat').reshape([20001,1,NX])
print(data.shape)
data = torch.from_numpy(data).float()
train_data_input  = Variable(data[10000:10100].reshape((-1,1,NX))).cuda()
train_data_output = Variable(data[10001:10101].reshape((-1,1,NX))).cuda()

loss = 1.0
loss_base, epoch = 10000.0, -1
while loss > 1.0e-15:
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
        torch.save(Network, 'CNN-L22-100.net')
