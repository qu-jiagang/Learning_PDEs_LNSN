import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from math import *


layer = 2
node = 32
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d( 1, node*2, 3, 1, 0),
            nn.GELU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(node*2,  1, 1, 1, 0, bias=False),
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
u = np.fromfile('dataset/2d-Allen-Cahn-0.dat').reshape([1100,1,NX,NY])[99:]
data = u

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

    print('Case 6:', epoch, loss)

    if epoch > 0 and loss < loss_base:
        loss_base = loss
        torch.save(dcign, 'CNN-layer'+str(layer)+'-nodes'+str(node)+'.net')