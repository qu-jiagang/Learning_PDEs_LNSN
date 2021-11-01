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

NX = NY = 128

u1 = np.fromfile('dataset/2d_headting_sdt_1.dat').reshape([3001, NX, NY])
r1 = np.fromfile('recon.dat').    reshape([3001, NX, NY])
r2 = np.fromfile('recon_ldt.dat').reshape([301, NX, NY])

fraction = 0.05
plt.figure(figsize=(10,6))

c = plt.cm.RdBu
for i in range(4):
    min = np.min(u1[i*1000])
    max = np.max(u1[i*1000])

    plt.subplot2grid((3,4),(0,i))
    plt.imshow(u1[i*1000],vmin=min,vmax=max,cmap=c)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.title('t = '+str(round(i*1,0))+' s')

    plt.subplot2grid((3,4),(1,i))
    plt.imshow(r1[i*1000],vmin=min,vmax=max,cmap=c)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.title('t = '+str(round(i*1,0))+' s')

    plt.subplot2grid((3,4),(2,i))
    plt.imshow(r2[i*100],vmin=min,vmax=max,cmap=c)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.title('t = '+str(round(i*1,0))+' s')

plt.tight_layout()
plt.show()