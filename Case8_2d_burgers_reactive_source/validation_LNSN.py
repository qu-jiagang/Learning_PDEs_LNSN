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


# layer = 2
# node  = 64
# Network = torch.load('LNSN-layer'+str(layer)+'-nodes'+str(node)+'.net')
# Network.eval()

Ts = 301

for layer in range(1,4):
    for node in [8,16,32,64]:
        error = np.zeros([1000,301])
        Network = torch.load('Network/LNSN-layer' + str(layer) + '-nodes' + str(node) + '.net')
        Network.eval()
        for no in range(1,1001):
            recon_x_sum = np.zeros((Ts,2,128,128))
            u = np.fromfile('dataset/dataset_u_'+str(no)+'.dat').reshape([301,1,128,128])
            v = np.fromfile('dataset/dataset_v_'+str(no)+'.dat').reshape([301,1,128,128])
            data = np.hstack([u,v])
            recon_x_sum[0] = data[0]
            Initial = data[0].reshape([1,2,128,128])
            input_ = Initial
            for i in range(Ts):
                input = Variable(torch.from_numpy(input_).float()).cuda()
                x_reconst = Network(input)
                input_ = x_reconst.cpu().data.numpy()
                recon_x_sum[i+1:i+2] = x_reconst.cpu().data.numpy()
                del x_reconst

            error[no-1] = np.sum((recon_x_sum-data)**2, axis=(1,2,3))/np.sum(data**2,axis=(1,2,3))
            print(layer, node, no, np.max(error[no-1]))

        np.savetxt('error/error-lnsn-' + str(layer) + '-' + str(node) + '.txt', error)