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
Nx = 128

data_11 = np.fromfile('dataset/dataset_1_0.0025.dat').reshape([4000, Nx])
data_12 = np.fromfile('dataset/dataset_1_0.01.dat'  ).reshape([1000, Nx])
data_21 = np.fromfile('dataset/dataset_2_0.0025.dat').reshape([4000, Nx])
data_22 = np.fromfile('dataset/dataset_2_0.01.dat'  ).reshape([1000, Nx])
data_31 = np.fromfile('dataset/dataset_3_0.0025.dat').reshape([4000, Nx])
data_32 = np.fromfile('dataset/dataset_3_0.01.dat'  ).reshape([1000, Nx])

reco_11 = np.fromfile('reconst/recon_1_0.0025.dt').reshape([4000, Nx])
reco_12 = np.fromfile('reconst/recon_1_0.01.dt'  ).reshape([1000, Nx])
reco_21 = np.fromfile('reconst/recon_2_0.0025.dt').reshape([4000, Nx])
reco_22 = np.fromfile('reconst/recon_2_0.01.dt'  ).reshape([1000, Nx])
reco_31 = np.fromfile('reconst/recon_3_0.0025.dt').reshape([4000, Nx])
reco_32 = np.fromfile('reconst/recon_3_0.01.dt'  ).reshape([1000, Nx])

reco_13 = np.fromfile('reconst/recon_K5_1_0.01.dt').reshape([1000, Nx])
reco_23 = np.fromfile('reconst/recon_K5_2_0.01.dt').reshape([1000, Nx])
reco_33 = np.fromfile('reconst/recon_K5_3_0.01.dt').reshape([1000, Nx])

x = np.linspace(0,1,128)
t = np.linspace(0,10,1000)

c1 = 'black'
c2 = plt.cm.tab10(0)
linewidth = 2
fig = plt.figure(figsize=(10,8))

plt.subplot(331)
plt.plot(x,data_11[1::800].T,c=c1,linewidth=linewidth)
plt.plot(x,reco_11[::800].T,'--',c=c2,linewidth=linewidth)
plt.xlim(0,1)
plt.xticks(fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)
plt.xlabel('$x$',fontsize=fontsize)
plt.ylabel('$u$',fontsize=fontsize)

plt.subplot(332)
plt.plot(x,data_21[1::800].T,c=c1,linewidth=linewidth)
plt.plot(x,reco_21[::800].T,'--',c=c2,linewidth=linewidth)
plt.xlim(0,1)
plt.xticks(fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)
plt.xlabel('$x$',fontsize=fontsize)
plt.ylabel('$u$',fontsize=fontsize)

plt.subplot(333)
plt.plot(x,data_31[1::800].T,c=c1,linewidth=linewidth)
plt.plot(x,reco_31[::800].T,'--',c=c2,linewidth=linewidth)
plt.xlim(0,1)
plt.xticks(fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)
plt.xlabel('$x$',fontsize=fontsize)
plt.ylabel('$u$',fontsize=fontsize)

plt.subplot(334)
plt.plot(x,data_12[1::200].T,c=c1,linewidth=linewidth)
plt.plot(x,reco_12[::200].T,'--',c=c2,linewidth=linewidth)
plt.xlim(0,1)
plt.xticks(fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)
plt.xlabel('$x$',fontsize=fontsize)
plt.ylabel('$u$',fontsize=fontsize)

plt.subplot(335)
plt.plot(x,data_22[1::200].T,c=c1,linewidth=linewidth)
plt.plot(x,reco_22[::200].T,'--',c=c2,linewidth=linewidth)
plt.xlim(0,1)
plt.xticks(fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)
plt.xlabel('$x$',fontsize=fontsize)
plt.ylabel('$u$',fontsize=fontsize)

plt.subplot(336)
plt.plot(x,data_32[1::200].T,c=c1,linewidth=linewidth)
plt.plot(x,reco_32[::200].T,'--',c=c2,linewidth=linewidth)
plt.xlim(0,1)
plt.xticks(fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)
plt.xlabel('$x$',fontsize=fontsize)
plt.ylabel('$u$',fontsize=fontsize)

plt.subplot(337)
plt.plot(x,data_12[1::200].T,c=c1,linewidth=linewidth)
plt.plot(x,reco_13[::200].T,'--',c=c2,linewidth=linewidth)
plt.xlim(0,1)
plt.xticks(fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)
plt.xlabel('$x$',fontsize=fontsize)
plt.ylabel('$u$',fontsize=fontsize)

plt.subplot(338)
plt.plot(x,data_22[1::200].T,c=c1,linewidth=linewidth)
plt.plot(x,reco_23[::200].T,'--',c=c2,linewidth=linewidth)
plt.xlim(0,1)
plt.xticks(fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)
plt.xlabel('$x$',fontsize=fontsize)
plt.ylabel('$u$',fontsize=fontsize)

plt.subplot(339)
plt.plot(x,data_32[1::200].T,c=c1,linewidth=linewidth)
plt.plot(x,reco_33[::200].T,'--',c=c2,linewidth=linewidth)
plt.xlim(0,1)
plt.xticks(fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)
plt.xlabel('$x$',fontsize=fontsize)
plt.ylabel('$u$',fontsize=fontsize)

plt.tight_layout()
plt.show()