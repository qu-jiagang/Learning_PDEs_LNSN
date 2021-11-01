import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class Net(nn.Module):
    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x = torch.cat((x1,x2),dim=1)
        x = self.layer3(x)
        return x


Network = torch.load('LNSN.net')
Network.eval()
print(Network)

NX = 128
burgers = np.fromfile('dataset/Allen-Cahn_3.dat').reshape([-1, NX])
Ts = np.size(burgers,0)
Initial = burgers[0].reshape([1,1,NX])
burgers_ = torch.from_numpy(burgers).float().cuda()
recon_x_sum = np.zeros([Ts,NX])
recon_x_sum[0] = burgers[0]

input_ = Initial
for i in range(Ts-1):
    input = Variable(torch.from_numpy(input_).float()).cuda()
    x_reconst = Network(input)
    input_ = x_reconst.cpu().data.numpy()
    recon_x_sum[i+1:i+2] = input_

    del x_reconst

recon_x_sum.tofile('recon/recon_lnsn_3.dat')
