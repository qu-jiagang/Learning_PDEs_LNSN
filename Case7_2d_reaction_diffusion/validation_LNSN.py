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
        x = self.layer3(x)
        return  x


Ts = 1001
NX=NY=128
recon_x_sum = np.zeros((Ts,1,NX,NY))
error = np.zeros([1000,Ts])

Network = torch.load('LNSN-layer1-nodes32.net')
Network.eval()

for no in range(5,1001):
    data = np.fromfile('dataset/2d-Allen-Cahn-'+str(no)+'.dat').reshape([1100,1,NX,NY])[99:]
    recon_x_sum = np.zeros((Ts,1,NX,NY))
    Initial = data[0].reshape([1,1,NX,NY])
    input_ = Initial
    recon_x_sum[0] = data[0]
    for i in range(Ts-1):
        input = Variable(torch.from_numpy(input_).float()).cuda()
        x_reconst = Network(input)
        input_ = x_reconst.cpu().data.numpy()
        recon_x_sum[i+1:i+2] = x_reconst.cpu().data.numpy()
        del x_reconst

    recon_x_sum.tofile('recon_lnsn.dat')
    exit()

    for i in range(Ts):
        error[no-1,i] = np.sum((recon_x_sum[i]-data[i])**2)/np.sum((data[i])**2)

    print(no, np.max(error[no-1]))

    # for i in range(4):
    #     min = np.min(data[i*250,0])
    #     max = np.max(data[i * 250, 0])
    #     plt.subplot(241+i)
    #     plt.imshow(data[i*250,0],vmin=min,vmax=max)
    #     plt.subplot(245+i)
    #     plt.imshow(recon_x_sum[i*250,0],vmin=min,vmax=max)
    # plt.show()

np.savetxt('error-lnsn.txt', error)
