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
            nn.Conv1d(1, 128*2, 3, 1, 0),
            nn.GELU(),
            nn.Conv1d(128*2, 1, 1, 1, 0, bias=False),
            nn.ConstantPad1d((1, 0), -1),
            nn.ConstantPad1d((0, 1),  1),
        )

    def forward(self, x):
        x = self.layer1(x)
        return x


def loss_func(determination, x):
    loss = F.mse_loss(determination, x, reduction='sum')
    return loss


Network = Net().cuda()

optimizer = optim.Adam(Network.parameters(), lr=1.0e-4)

data = np.fromfile('Allen-Cahn_1.dat').reshape([1, 10000, 128])
data = torch.from_numpy(data).float()
train_data_input  = Variable(data[:,0:10].reshape((-1,1,128))).cuda()
train_data_output = Variable(data[:,1:11].reshape((-1,1,128))).cuda()

loss_base, epoch = 1.0, -1
while epoch < 1000000:
    epoch = epoch + 1

    x_reconst = Network(train_data_input)
    loss = loss_func(x_reconst, train_data_output)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss = loss.cpu().data.numpy()

    print('Case 3:', epoch, loss)

    if epoch > 0 and loss < loss_base:
        loss_base = loss
        torch.save(Network, 'CNN.net')
