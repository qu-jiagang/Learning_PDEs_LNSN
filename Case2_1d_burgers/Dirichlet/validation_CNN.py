import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


class Net(nn.Module):
    def forward(self, x):
        x = self.layer1(x)
        return x


Network = torch.load('CNN.net')
Network.eval()
print(Network)

for no in range(3):
    print(no)
    Ts = 1000
    NX = 128

    recon = np.zeros((Ts,NX))
    burgers = np.fromfile('../dataset/1d_burgers_Dirichlet_'+str(no)+'.dat').reshape([10001, NX])[::10]
    Initial = burgers[0].reshape([1,1,NX])

    input_ = Initial
    for i in range(Ts):
        input = Variable(torch.from_numpy(input_).float()).cuda()
        x_reconst = Network(input)
        input_ = x_reconst.cpu().data.numpy()
        recon[i:i+1] = x_reconst.cpu().data.numpy()
        del x_reconst

    recon.tofile('reconst/recon_CNN_'+str(no)+'.dat')

    # plot
    plt.figure()
    plt.plot(burgers[1::200,:].T)
    plt.plot(recon[::200,:].T,'--')

plt.show()
