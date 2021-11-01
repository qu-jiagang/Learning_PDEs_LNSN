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
        x = self.layer1(x)
        return x


Nx = 128       # number of grids
T  = 10        # total Time
dt = 0.001     # Time step: small dt=0.001 large dt=0.0025
Ts = int(T/dt) # total steps
mu = 0.01      # parameter

Network = torch.load('CNN_'+str(dt)+'.net')
Network.eval()

for para in Network.parameters():
    print(para)
# exit()

dataset_1 = np.fromfile('dataset/dataset_1_'+str(dt)+'.dat').reshape([Ts,Nx])
dataset_2 = np.fromfile('dataset/dataset_2_'+str(dt)+'.dat').reshape([Ts,Nx])
dataset_3 = np.fromfile('dataset/dataset_3_'+str(dt)+'.dat').reshape([Ts,Nx])

recon_1 = np.zeros((Ts,Nx))
recon_1[0] = dataset_1[0]
recon_2 = np.zeros((Ts,Nx))
recon_2[0] = dataset_2[0]
recon_3 = np.zeros((Ts,Nx))
recon_3[0] = dataset_3[0]

input_1 = recon_1[0].reshape([1,1,Nx])
input_2 = recon_2[0].reshape([1,1,Nx])
input_3 = recon_3[0].reshape([1,1,Nx])
for i in range(Ts-1):
    input_1 = Variable(torch.from_numpy(input_1).float()).cuda()
    input_2 = Variable(torch.from_numpy(input_2).float()).cuda()
    input_3 = Variable(torch.from_numpy(input_3).float()).cuda()
    x_reconst_1 = Network(input_1)
    x_reconst_2 = Network(input_2)
    x_reconst_3 = Network(input_3)
    input_1 = x_reconst_1.cpu().data.numpy()
    input_2 = x_reconst_2.cpu().data.numpy()
    input_3 = x_reconst_3.cpu().data.numpy()

    recon_1[i + 1:i + 2] = x_reconst_1.cpu().data.numpy()
    recon_2[i + 1:i + 2] = x_reconst_2.cpu().data.numpy()
    recon_3[i + 1:i + 2] = x_reconst_3.cpu().data.numpy()

    del x_reconst_1,x_reconst_2,x_reconst_3

recon_1.tofile('reconst/recon_1'+str(dt)+'.dt')
recon_2.tofile('reconst/recon_2'+str(dt)+'.dt')
recon_3.tofile('reconst/recon_3'+str(dt)+'.dt')

deltaT = int(2/dt)
plt.figure(figsize=(10,3))
plt.subplot(131)
plt.plot(dataset_1[::deltaT].T)
plt.plot(recon_1[::deltaT].T,'--',)
plt.subplot(132)
plt.plot(dataset_2[::deltaT].T)
plt.plot(recon_2[::deltaT].T,'--')
plt.subplot(133)
plt.plot(dataset_3[::deltaT].T)
plt.plot(recon_3[::deltaT].T,'--')
plt.show()
