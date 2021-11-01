import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from math import *


layer = 3
node = 64

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(   2, node, 3, 1, 1, bias=False, padding_mode='circular'),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(   2, node, 3, 1, 1, padding_mode='circular'),
            nn.GELU(),
            nn.Conv2d(node, node, 3, 1, 1, padding_mode='circular'),
            nn.GELU(),
            nn.Conv2d(node, node, 3, 1, 1, padding_mode='circular'),
            nn.GELU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(node*2,  2, 1, 1, 0, bias=False, padding_mode='circular'),
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x = torch.cat((x1,x2),1)
        x_reconst = self.layer3(x)
        return  x_reconst


def loss_func(determination, x):
    BCE = F.mse_loss(determination, x, reduction='sum')
    return BCE


dcign = Net().cuda()

optimizer = optim.Adam(dcign.parameters(), lr=1.0e-4)

u = np.fromfile('dataset/dataset_u_0.dat').reshape([301,1,128,128])
v = np.fromfile('dataset/dataset_v_0.dat').reshape([301,1,128,128])
data = np.hstack([u,v])

data = torch.from_numpy(data).float()
train_data_input  = Variable(data[0:1].reshape((1,2,128,128))).cuda()
train_data_output = Variable(data[1:2].reshape((1,2,128,128))).cuda()

_loss, loss_base, epoch = 1.0, 1000000.0, -1

loss = 1000
while epoch < 1000000:
    epoch = epoch + 1

    x_reconst = dcign(train_data_input)
    loss = loss_func(x_reconst, train_data_output)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss = loss.cpu().data.numpy()

    print('Case 7:',epoch, loss)

    if epoch > 0 and loss < loss_base:
        loss_base = loss
        torch.save(dcign, 'LNSN-layer'+str(layer)+'-nodes'+str(node)+'.net')