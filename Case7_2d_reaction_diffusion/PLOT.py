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
no = 4
data = np.fromfile('dataset/2d-Allen-Cahn-5.dat').reshape([1100,NX,NY])[99:]
CNN  = np.fromfile('recon_cnn.dat').reshape([1001, 128, 128])
LNSN = np.fromfile('recon_lnsn.dat').reshape([1001, 128, 128])

c = plt.cm.RdBu
plt.figure(figsize=(12,6))
for i in range(5):
    index = 10**(-3)*250
    vmax = np.max(data[i*250])
    vmin = np.min(data[i*250])
    plt.subplot2grid((3, 5), (0, i))
    plt.imshow(data[i*250], vmin=-1, vmax=1, cmap=c)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.title('t = ' + str(round(i * index, 2)) + ' s')

    plt.subplot2grid((3, 5), (1, i))
    plt.imshow(LNSN[i*250], vmin=-1, vmax=1, cmap=c)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.title('t = ' + str(round(i * index, 2)) + ' s')

    plt.subplot2grid((3, 5), (2, i))
    plt.imshow(CNN[i*250], vmin=-1, vmax=1, cmap=c)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.title('t = '+str(round(i * index,2))+' s')

    plt.subplot2grid((3, 5), (2, i))
    plt.imshow(CNN[i*250], vmin=-1, vmax=1, cmap=c)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.title('t = '+str(round(i * index,2))+' s')

plt.tight_layout()
plt.show()