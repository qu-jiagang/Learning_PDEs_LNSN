import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer_x3 = nn.Sequential(
            nn.Conv1d(1, 1, 3, 1, 0, bias=False),
        )

        self.layer_x5 = nn.Sequential(
            nn.Conv1d(1, 1, 5, 1, 0, bias=False),
        )

    def forward(self, x):
        x3 = self.layer_x3(x)
        x5 = self.layer_x5(x)
        x5 = torch.cat((x3[:,:,0:1],x5,x3[:,:,-1:]),dim=2)
        x5 = F.pad(x5,(1,1),mode='constant',value=0)
        x3 = F.pad(x3,(1,1),mode='constant',value=0)
        return x3, x5


def loss_func(determination, x3, x5):
    loss_x3 = F.mse_loss(determination, x3, reduction='sum')
    loss_x5 = F.mse_loss(determination, x5, reduction='sum')
    loss = loss_x3 + loss_x5
    return loss


Nx = 128        # number of grids
T  = 10         # total Time
dt = 0.01
Ts = int(T/dt)  # total steps

data = np.fromfile('dataset/dataset_1_'+str(dt)+'.dat').reshape([Ts,1,Nx])

Network = Net().cuda()
optimizer = optim.Adam(Network.parameters(), lr=1.0e-4)

data = torch.from_numpy(data).float()
train_data_input  = Variable(data[0:10].reshape((-1,1,128))).cuda()
train_data_output = Variable(data[1:11].reshape((-1,1,128))).cuda()

loss_base, epoch = 1.0, -1
while epoch < 1000000:
    epoch = epoch + 1
    x3, x5 = Network(train_data_input)
    loss = loss_func(train_data_output, x3, x5)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss = loss.cpu().data.numpy()

    print(epoch, loss)

    if epoch > 0 and loss < loss_base:
        loss_base = loss
        torch.save(Network, 'CNN5_'+str(dt)+'.net')

