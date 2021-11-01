import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from math import *
from mpl_toolkits.mplot3d import Axes3D


class Net(nn.Module):
    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x = torch.cat((x1,x2),1)
        x_reconst = self.layer3(x)
        return  x_reconst


Network = torch.load('LNSN.net')
Network.eval()

Ts = 1001
NX=NY=64
recon_x_sum = np.zeros((Ts,1,NX,NY))
u = np.fromfile('2d-KS_dataset.dat').reshape([40001,1,NX,NY])[20000:20000+Ts:1]
data = u

Initial = data[0].reshape([1,1,NX,NY])
input_ = Initial
recon_x_sum[0:1] = Initial
for i in range(Ts-1):
    input = Variable(torch.from_numpy(input_).float()).cuda()
    x_reconst = Network(input)
    input_ = x_reconst.cpu().data.numpy()
    recon_x_sum[i+1:i+2] = x_reconst.cpu().data.numpy()
    del x_reconst
    print(i)
    
recon_x_sum.tofile('recon_LNSN_1.dat')

