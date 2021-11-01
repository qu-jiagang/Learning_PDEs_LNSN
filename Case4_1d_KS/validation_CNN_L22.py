import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


class Net(nn.Module):
    def forward(self, x):
        x = self.layer1(x)
        return x


Network = torch.load('CNN-L22-100.net')
Network.eval()

Ts = 10000-100
NX = 64
data = np.fromfile('1d_ks_L22.dat').reshape([20001,1,NX])[10100:]
recon_x_sum = np.zeros((Ts,1,NX))

Initial = data[0].reshape([1,1,NX])
input_ = Initial
for i in range(Ts):
    input = Variable(torch.from_numpy(input_).float()).cuda()
    x_reconst = Network(input)
    input_ = x_reconst.cpu().data.numpy()
    recon_x_sum[i:i+1] = x_reconst.cpu().data.numpy()
    del x_reconst
    print(i)
recon_x_sum.tofile('recon_CNN_L22.dat')
