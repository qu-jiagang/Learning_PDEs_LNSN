import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


layer = 1
node = 32

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(     2, node*2, 3, 1, 1, padding_mode='circular'),
            nn.GELU(),
            # nn.Conv2d(node*2, node*2, 3, 1, 1, padding_mode='circular'),
            # nn.GELU(),
            # nn.Conv2d(node*2, node*2, 3, 1, 1, padding_mode='circular'),
            # nn.GELU(),
            nn.Conv2d(node*2,      2, 1, 1, 0, bias=False, padding_mode='circular'),
        )

    def forward(self, x):
        x = self.layer1(x)
        return x


def loss_func(determination, x):
    loss = F.mse_loss(determination, x, reduction='sum')
    return loss


dcign = Net().cuda()

optimizer = optim.Adam(dcign.parameters(), lr=1.0e-4)

u = np.fromfile('dataset/dataset_u_0.dat').reshape([301,1,128,128])[::10]
v = np.fromfile('dataset/dataset_v_0.dat').reshape([301,1,128,128])[::10]
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
        torch.save(dcign, 'CNN-layer'+str(layer)+'-nodes'+str(node)+'.net')
