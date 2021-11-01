import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class Net(nn.Module):
    def forward(self, x):
        x = self.layer1(x)
        return x


Ts = 301
error = np.zeros([1000,Ts])
for layer in range(1,2):
    for node in [32]:
        Network = torch.load('Network/CNN-layer' + str(layer) + '-nodes' + str(node) + '.net')
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

        np.savetxt('error/error-cnn-' + str(layer) + '-' + str(node) + '.txt', error)
