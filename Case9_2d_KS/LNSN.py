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
            nn.Conv2d(1,    node, 7, 1, 3, bias=False, padding_mode='circular'),
            nn.Conv2d(node, node, 7, 1, 3, bias=False, padding_mode='circular'),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(1,    node, 7, 1, 3, padding_mode='circular'),
            nn.GELU(),
            nn.Conv2d(node, node, 7, 1, 3, padding_mode='circular'),
            nn.GELU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(node+node, 1, 1, 1, 0, bias=False),
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x = torch.cat((x1,x2),1)
        x_reconst = self.layer3(x)
        return  x_reconst


def loss_func(determination, x):
    loss = F.mse_loss(determination, x)
    return loss


dcign = Net().cuda()

optimizer = optim.Adam(dcign.parameters(), lr=1.0e-5)

NX=NY=64
u = np.fromfile('2d-KS_dataset.dat').reshape([40001,1,NX,NY])[20000::]
data = u

data = torch.from_numpy(data).float()
train_data_input  = Variable(data[0:10].reshape((-1,1,NX,NY))).cuda()
train_data_output = Variable(data[1:11].reshape((-1,1,NX,NY))).cuda()

loss_base, epoch = 1.0e-8, -1
while epoch < 2000000:
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
        torch.save(dcign, 'LNSN.net')
