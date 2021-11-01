import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

plt.rc('font',family='Times New Roman')
fontsize_ = 12
fontsize = 16

config = {
    "font.size":12,
    "mathtext.fontset":'stix',
}
rcParams.update(config)

NX = 64

data = np.fromfile('1d_ks_L22.dat').reshape([20001,NX])[10000:14001]

CNN  = np.fromfile('recon_CNN_L22.dat' ).reshape([-1,NX])[:4001-100]
LNSN = np.fromfile('recon_LNSN_L22.dat').reshape([-1,NX])[:4001-100]
CNN  = np.vstack((data[:100],CNN))
LNSN = np.vstack((data[:100],LNSN))

error_CNN = np.sum((CNN -data)**2,axis=1)/np.sum(data**2,axis=1)
error_LNSN = np.sum((LNSN-data)**2,axis=1)/np.sum(data**2,axis=1)
print(error_CNN[1600])
print(error_LNSN[1600])
plt.plot(error_CNN)
plt.plot(error_LNSN)
plt.show()

c = plt.cm.RdBu
x = np.array([0,50,100,150,200])
t = np.array(x/2,dtype=int)
vmin = np.min(data)
vmax = np.max(data)

plt.figure(figsize=(9,5))
plt.subplot2grid((3,2),(0,0))
plt.title('LNSN')
plt.imshow(data[::20].T,cmap=c,vmin=vmin,vmax=vmax)
plt.xlabel('$t$',fontsize=fontsize)
plt.ylabel('$x$',fontsize=fontsize)
plt.xticks(x,t,fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)
plt.vlines(100/20,0,63,color='black')
plt.colorbar()

plt.subplot2grid((3,2),(1,0))
plt.imshow(LNSN[::20].T,cmap=c,vmin=vmin,vmax=vmax)
plt.xlabel('$t$',fontsize=fontsize)
plt.ylabel('$x$',fontsize=fontsize)
plt.xticks(x,t,fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)
plt.vlines(100/20,0,63,color='black')
plt.colorbar()

plt.subplot2grid((3,2),(2,0))
plt.imshow(LNSN[::20].T-data[::20].T,cmap=c,vmin=vmin,vmax=vmax)
plt.xlabel('$t$',fontsize=fontsize)
plt.ylabel('$x$',fontsize=fontsize)
plt.xticks(x,t,fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)
plt.vlines(100/20,0,63,color='black')
plt.colorbar()


plt.subplot2grid((3,2),(0,1))
plt.title('CNN')
plt.imshow(data[::20].T,cmap=c,vmin=vmin,vmax=vmax)
plt.xlabel('$t$',fontsize=fontsize)
plt.ylabel('$x$',fontsize=fontsize)
plt.xticks(x,t,fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)
plt.vlines(100/20,0,63,color='black')
plt.colorbar()

plt.subplot2grid((3,2),(1,1))
plt.imshow(CNN[::20].T ,cmap=c,vmin=vmin,vmax=vmax)
plt.xlabel('$t$',fontsize=fontsize)
plt.ylabel('$x$',fontsize=fontsize)
plt.xticks(x,t,fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)
plt.vlines(100/20,0,63,color='black')
plt.colorbar()

plt.subplot2grid((3,2),(2,1))
plt.imshow(CNN[::20].T-data[::20].T,cmap=c,vmin=vmin,vmax=vmax)
plt.xlabel('$t$',fontsize=fontsize)
plt.ylabel('$x$',fontsize=fontsize)
plt.xticks(x,t,fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)
plt.vlines(100/20,0,63,color='black')
plt.colorbar()

plt.tight_layout()
plt.show()