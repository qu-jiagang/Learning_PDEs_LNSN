import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from math import *
from matplotlib import rcParams


rcParams.update({
    "font.size":12,
    "mathtext.fontset":'stix',
    "font.family":"serif",
})
plt.rc('font',family='Times New Roman')
fontsize_ = 12
fontsize = 16

def error(x,y):
    return np.sum((x-y)**2,axis=1)/np.sum(x**2,axis=1)


data_D1 = np.fromfile('dataset/1d_burgers_Dirichlet_2.dat').reshape([10001, 128])[::10]
data_D2 = np.fromfile('dataset/1d_burgers_Dirichlet_4.dat').reshape([10001, 128])[::10]
data_N1 = np.fromfile('dataset/1d_burgers_Neumann_1.dat').reshape([10001, 128])[::10]
data_N2 = np.fromfile('dataset/1d_burgers_Neumann_2.dat').reshape([10001, 128])[::10]
data_P1 = np.fromfile('dataset/1d_burgers_Periodic_2.dat').reshape([10001, 402])[::10]
data_P2 = np.fromfile('dataset/1d_burgers_Periodic_4.dat').reshape([10001, 402])[::10]

CNN_D1 = np.fromfile('Dirichlet/reconst/recon_CNN_2.dat').reshape([1000, 128])
CNN_D2 = np.fromfile('Dirichlet/reconst/recon_CNN_4.dat').reshape([1000, 128])
CNN_N1 = np.fromfile('Neumnn/reconst/recon_CNN_1.dat').reshape([1000, 128])
CNN_N2 = np.fromfile('Neumnn/reconst/recon_CNN_2.dat').reshape([1000, 128])
CNN_P1 = np.fromfile('Periodic/reconst/recon_CNN_2.dat').reshape([1000, 402])
CNN_P2 = np.fromfile('Periodic/reconst/recon_CNN_4.dat').reshape([1000, 402])

LNSN_D1 = np.fromfile('Dirichlet/reconst/recon_LNSN_2.dat').reshape([1000, 128])
LNSN_D2 = np.fromfile('Dirichlet/reconst/recon_LNSN_4.dat').reshape([1000, 128])
LNSN_N1 = np.fromfile('Neumnn/reconst/recon_LNSN_1.dat').reshape([1000, 128])
LNSN_N2 = np.fromfile('Neumnn/reconst/recon_LNSN_2.dat').reshape([1000, 128])
LNSN_P1 = np.fromfile('Periodic/reconst/recon_LNSN_2.dat').reshape([1000, 402])
LNSN_P2 = np.fromfile('Periodic/reconst/recon_LNSN_4.dat').reshape([1000, 402])

print((error(data_D1[1:],LNSN_D1)[-1]+error(data_D2[1:],LNSN_D2)[-1])/2)
print((error(data_D1[1:],CNN_D1 )[-1]+error(data_D2[1:],CNN_D2 )[-1])/2)
print((error(data_N1[1:],LNSN_N1)[-1]+error(data_N2[1:],LNSN_N2)[-1])/2)
print((error(data_N1[1:],CNN_N1 )[-1]+error(data_N2[1:],CNN_N2 )[-1])/2)
print((error(data_P1[1:],LNSN_P1)[-1]+error(data_P2[1:],LNSN_P2)[-1])/2)
print((error(data_P1[1:],CNN_P1 )[-1]+error(data_P2[1:],CNN_P2 )[-1])/2)

x = np.linspace(-1,1,128)
xP = np.linspace(-pi,pi,402)
t = np.linspace(0,10,1000)

c1 = 'black'
c2 = plt.cm.tab10(0)
linewidth = 2
fig = plt.figure(figsize=(10,16))

plt.subplot2grid((6,3),(0,0))
plt.plot(x,data_D1[1::200].T,c=c1,linewidth=linewidth)
plt.plot(x,LNSN_D1[::200].T,'--',c=c2,linewidth=linewidth)
plt.xlim(-1,1)
plt.title('LNSN',fontsize=fontsize)
plt.xticks(fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)

plt.subplot2grid((6,3),(0,1))
plt.plot(x,data_D1[1::200].T,c=c1,linewidth=linewidth)
plt.plot(x,CNN_D1[::200].T,'--',c=c2,linewidth=linewidth)
plt.xlim(-1,1)
plt.title('CNN',fontsize=fontsize)
plt.xticks(fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)

plt.subplot2grid((6,3),(0,2))
plt.plot(t,error(data_D1[1:],LNSN_D1),label='LNSN',linewidth=linewidth)
plt.plot(t,error(data_D1[1:],CNN_D1),label='CNN',linewidth=linewidth)
plt.legend(fontsize=fontsize)
plt.xlim(0,10)
plt.title('Error',fontsize=fontsize)
plt.xticks(fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)

plt.subplot2grid((6,3),(1,0))
plt.plot(x,data_D2[::200].T,c=c1,linewidth=linewidth)
plt.plot(x,LNSN_D2[::200].T,'--',c=c2,linewidth=linewidth)
plt.xlim(-1,1)
plt.xticks(fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)

plt.subplot2grid((6,3),(1,1))
plt.plot(x,data_D2[::200].T,c=c1,linewidth=linewidth)
plt.plot(x,CNN_D2[::200].T,'--',c=c2,linewidth=linewidth)
plt.xlim(-1,1)
plt.xticks(fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)

plt.subplot2grid((6,3),(1,2))
plt.plot(t,error(data_D2[1:],LNSN_D2),label='LNSN',linewidth=linewidth)
plt.plot(t,error(data_D2[1:],CNN_D2),label='CNN',linewidth=linewidth)
plt.legend(fontsize=fontsize)
plt.xlim(0,10)
plt.xticks(fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)

plt.subplot2grid((6,3),(2,0))
plt.plot(x,data_N1[::200].T,c=c1,linewidth=linewidth)
plt.plot(x,LNSN_N1[::200].T,'--',c=c2,linewidth=linewidth)
plt.xlim(-1,1)
plt.xticks(fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)

plt.subplot2grid((6,3),(2,1))
plt.plot(x,data_N1[::200].T,c=c1,linewidth=linewidth)
plt.plot(x,CNN_N1[::200].T,'--',c=c2,linewidth=linewidth)
plt.xlim(-1,1)
plt.xticks(fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)

plt.subplot2grid((6,3),(2,2))
plt.plot(t,error(data_N1[1:],LNSN_N1),label='LNSN',linewidth=linewidth)
plt.plot(t,error(data_N1[1:],CNN_N1),label='CNN',linewidth=linewidth)
plt.legend(fontsize=fontsize)
plt.xlim(0,10)
plt.xticks(fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)

plt.subplot2grid((6,3),(3,0))
plt.plot(x,data_N2[::200].T,c=c1,linewidth=linewidth)
plt.plot(x,LNSN_N2[::200].T,'--',c=c2,linewidth=linewidth)
plt.xlim(-1,1)
plt.xticks(fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)

plt.subplot2grid((6,3),(3,1))
plt.plot(x,data_N2[::200].T,c=c1,linewidth=linewidth)
plt.plot(x,CNN_N2[::200].T,'--',c=c2,linewidth=linewidth)
plt.xlim(-1,1)
plt.xticks(fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)

plt.subplot2grid((6,3),(3,2))
plt.plot(t,error(data_N2[1:],LNSN_N2),label='LNSN',linewidth=linewidth)
plt.plot(t,error(data_N2[1:],CNN_N2),label='CNN',linewidth=linewidth)
plt.legend(fontsize=fontsize)
plt.xlim(0,10)
plt.xticks(fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)

plt.subplot2grid((6,3),(4,0))
plt.plot(xP,data_P1[::200].T,c=c1,linewidth=linewidth)
plt.plot(xP,LNSN_P1[::200].T,'--',c=c2,linewidth=linewidth)
plt.xlim(-pi,pi)
plt.xticks(fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)

plt.subplot2grid((6,3),(4,1))
plt.plot(xP,data_P1[::200].T,c=c1,linewidth=linewidth)
plt.plot(xP,CNN_P1[::200].T,'--',c=c2,linewidth=linewidth)
plt.xlim(-pi,pi)
plt.xticks(fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)

plt.subplot2grid((6,3),(4,2))
plt.plot(t,error(data_P1[1:],LNSN_P1),label='LNSN',linewidth=linewidth)
plt.plot(t,error(data_P1[1:],CNN_P1),label='CNN',linewidth=linewidth)
plt.legend(fontsize=fontsize)
plt.xlim(0,10)
plt.xticks(fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)

plt.subplot2grid((6,3),(5,0))
plt.plot(xP,data_P2[::200].T,c=c1,linewidth=linewidth)
plt.plot(xP,LNSN_P2[::200].T,'--',c=c2,linewidth=linewidth)
plt.xlim(-pi,pi)
plt.xlabel('$x$',fontsize=fontsize)
plt.xticks(fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)

plt.subplot2grid((6,3),(5,1))
plt.plot(xP,data_P2[::200].T,c=c1,linewidth=linewidth)
plt.plot(xP,CNN_P2[::200].T,'--',c=c2,linewidth=linewidth)
plt.xlim(-pi,pi)
plt.xlabel('$x$',fontsize=fontsize)
plt.xticks(fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)

plt.subplot2grid((6,3),(5,2))
plt.plot(t,error(data_P2[1:],LNSN_P2),label='LNSN',linewidth=linewidth)
plt.plot(t,error(data_P2[1:],CNN_P2),label='CNN',linewidth=linewidth)
plt.legend(fontsize=fontsize)
plt.xlim(0,10)
plt.xlabel('$t$',fontsize=fontsize)
plt.xticks(fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)

plt.tight_layout()
# plt.savefig('1d_burgers.eps')
plt.show()

