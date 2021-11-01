import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from math import *


class Net(nn.Module):
    def forward(self, x):
        x = self.layer1(x)
        return x


Network = torch.load('LCNN.net')
Network.eval()
Network_ldt = torch.load('LCNN_ldt.net')
Network_ldt.eval()

# for para in Network.parameters():
#     Q = para.cpu().data.numpy()[0,0]
#     print(Q)
#     print(np.max(np.abs(np.linalg.eigvals(Q))))
# exit()

nu = 0.1
Ts1 = 3001
Ts2 = 301

NX=NY=128
recon     = np.zeros((Ts1,1,NX,NY))
recon_ldt = np.zeros((Ts2,1,NX,NY))
error     = np.zeros((1000,Ts1))
error_ldt = np.zeros((1000,Ts2))

for no in range(1,1001):
    data = np.fromfile('dataset/2d_headting_sdt_'+str(no)+'.dat').reshape([3001,1,NX,NY])

    input = Variable(torch.from_numpy(data[0].reshape([1,1,NX,NY])).float().cuda())
    recon_ldt[0] = data[0].reshape([1,1,NX,NY])
    recon[0] = data[0].reshape([1,1,NX,NY])
    for i in range(Ts1-1):
        input = Network(input)
        recon[i+1:i+2] = input.cpu().data.numpy()

    recon.tofile('recon.dat')

    input = Variable(torch.from_numpy(data[0].reshape([1,1,NX,NY])).float().cuda())
    recon_ldt[0] = data[0].reshape([1,1,NX,NY])
    for i in range(Ts2-1):
        input = Network_ldt(input)
        recon_ldt[i+1:i+2] = input.cpu().data.numpy()

    recon_ldt.tofile('recon_ldt.dat')

    exit()

    for i in range(Ts1-1):
        error[no-1, i+1] = np.sum((recon[i+1]-data[i+1])**2)/np.sum((data[i+1])**2)
    for i in range(Ts2-1):
        error_ldt[no-1, i+1] = np.sum((recon_ldt[i+1]-data[::10][i+1])**2)/np.sum((data[::10][i+1])**2)

    print(no-1, np.max(error[no-1]),np.max(error_ldt[no-1]))

np.savetxt('error.txt', error)
np.savetxt('error-ldt.txt', error_ldt)
