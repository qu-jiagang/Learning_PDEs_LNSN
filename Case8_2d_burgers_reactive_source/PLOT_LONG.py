import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


rcParams.update({
    "font.size":12,
    "mathtext.fontset":'stix',
    "font.family":"serif",
})
plt.rc('font',family='Times New Roman')
fontsize_ = 12
fontsize = 16

no = 1
u = np.fromfile('dataset/dataset_u_' + str(no) + '.dat').real.reshape([301, 128, 128])
v = np.fromfile('dataset/dataset_v_' + str(no) + '.dat').real.reshape([301, 128, 128])
CNN  = np.fromfile('recon_CNN.dat').reshape([301, 2, 128, 128])
LNSN = np.fromfile('recon_LNSN.dat').reshape([301, 2, 128, 128])

c = plt.cm.RdBu
plt.figure(figsize=(10,6))
for i in range(4):
    vmax = np.max(u[i*100])
    vmin = np.min(u[i*100])
    plt.suptitle('Dynamics of $u$')
    plt.subplot2grid((3, 4), (0, i))
    plt.imshow(u[i*100], vmin=vmin, vmax=vmax, cmap=c)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.title('t = '+str(round(i*1,0))+' s')

    plt.subplot2grid((3, 4), (1, i))
    plt.imshow(LNSN[i*100,0], vmin=vmin, vmax=vmax, cmap=c)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.title('t = '+str(round(i*1,0))+' s')

    plt.subplot2grid((3, 4), (2, i))
    plt.imshow(LNSN[i*100,0]-u[i*100], vmin=vmin, vmax=vmax, cmap=c)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.title('t = '+str(round(i*1,0))+' s')

plt.tight_layout()

plt.figure(figsize=(10, 6))
for i in range(4):
    vmax = np.max(v[i*100])
    vmin = np.min(v[i*100])
    plt.suptitle('Dynamics of $v$')
    plt.subplot2grid((3, 4), (0, i))
    plt.imshow(v[i*100], vmin=vmin, vmax=vmax, cmap=c)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.title('t = '+str(round(i*1,0))+' s')

    plt.subplot2grid((3, 4), (1, i))
    plt.imshow(LNSN[i*100,1], vmin=vmin, vmax=vmax, cmap=c)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.title('t = '+str(round(i*1,0))+' s')

    plt.subplot2grid((3, 4), (2, i))
    plt.imshow(LNSN[i*100,1]-v[i*100], vmin=vmin, vmax=vmax, cmap=c)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.title('t = '+str(round(i*1,0))+' s')

plt.tight_layout()
plt.show()