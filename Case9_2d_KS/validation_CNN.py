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
        x_reconst = self.layer1(x)
        return  x_reconst


Network = torch.load('CNN.net')
Network.eval()
print(Network)

for para in Network.parameters():
    print(para.shape)

Ts = 1001
NX=NY=64
recon_x_sum = np.zeros((Ts,1,NX,NY))
u = np.fromfile('testing.dat').reshape([20001,1,NX,NY])[:Ts]
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
# recon_x_sum.tofile('recon_CNN_1.dat')

error = np.mean((data-recon_x_sum)**2,axis=(1,2,3))/np.mean((data)**2,axis=(1,2,3))
plt.figure()
plt.plot(error)

plt.figure()
plt.plot(data[:Ts,0,-16,16])
plt.plot(recon_x_sum[:,0,-16,16],'--')

c = plt.cm.RdBu_r
plt.figure(figsize=(10,4))
for i in range(10):
    index = i*100
    vmin = np.min(data[index,0])
    vmax = np.max(data[index,0])
    # vmin = np.min(data)
    # vmax = np.max(data)
    plt.subplot2grid((3,10),(0,i))
    plt.imshow(data[index,0],cmap=c,vmin=vmin,vmax=vmax)
    plt.subplot2grid((3,10),(1,i))
    plt.imshow(recon_x_sum[index,0],cmap=c,vmin=vmin,vmax=vmax)
    plt.subplot2grid((3,10),(2,i))
    plt.imshow(data[index,0]-recon_x_sum[index,0],cmap=c,vmin=vmin,vmax=vmax)
plt.show()
