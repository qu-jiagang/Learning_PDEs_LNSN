import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class Net(nn.Module):
    def forward(self, x):
        x = self.layer1(x)
        return x


for no in range(3):
    print(no)
    Ts = 1000
    NX = 128

    Network = torch.load('CNN.net')
    Network.eval()

    recon = np.zeros((Ts,NX))
    burgers = np.fromfile('../dataset/1d_burgers_Neumann_'+str(no)+'.dat').reshape([10001, NX])[::10]
    Initial = burgers[0].reshape([1,1,NX])

    input_ = Initial
    for i in range(Ts):
        input = Variable(torch.from_numpy(input_).float()).cuda()
        x_reconst = Network(input)
        input_ = x_reconst.cpu().data.numpy()
        recon[i:i+1] = x_reconst.cpu().data.numpy()
        del x_reconst

    recon.tofile('reconst/recon_CNN_'+str(no)+'.dat')
