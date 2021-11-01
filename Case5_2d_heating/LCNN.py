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
            nn.Conv2d(1, 1, 3, 1, 1, bias=False, padding_mode='circular'),
        )

    def forward(self, x):
        x = self.layer1(x)
        return x


def loss_func(determination, x):
    BCE = F.mse_loss(determination, x, reduction='sum')
    return BCE


dcign = Net().cuda()

optimizer = optim.Adam(dcign.parameters(), lr=1.0e-4)

NX=NY=128
data = np.fromfile('dataset/2d_headting_sdt_0.dat').reshape([3001,1,NX,NY])[::10]

data = torch.from_numpy(data).float()
train_data_input  = Variable(data[0:1].reshape((-1,1,NX,NY))).cuda()
train_data_output = Variable(data[1:2].reshape((-1,1,NX,NY))).cuda()

loss_base, epoch = 100, -1
while epoch < 1000000:
    epoch = epoch + 1

    x_reconst = dcign(train_data_input)
    loss = loss_func(x_reconst, train_data_output)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss = loss.cpu().data.numpy()

    print(epoch, loss)

    if epoch > 0 and loss < loss_base:
        loss_base = loss
        torch.save(dcign, 'LCNN_ldt.net')