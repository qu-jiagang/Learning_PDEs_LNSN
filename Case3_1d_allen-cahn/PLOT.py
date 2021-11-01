import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams


rcParams.update({
    "font.size":12,
    "mathtext.fontset":'stix',
    "font.family":"serif",
})
plt.rc('font',family='Times New Roman')
fontsize_ = 12
fontsize = 16

NX = 128
data2 = np.fromfile('dataset/Allen-Cahn_2.dat').reshape([-1, NX])
data3 = np.fromfile('dataset/Allen-Cahn_3.dat').reshape([-1, NX])
print(data2.shape)
print(data3.shape)

recon_cnn_2 = np.fromfile('recon/recon_cnn_2.dat').reshape([-1, NX])
recon_cnn_3 = np.fromfile('recon/recon_cnn_3.dat').reshape([-1, NX])
recon_lnsn_2 = np.fromfile('recon/recon_lnsn_2.dat').reshape([-1, NX])
recon_lnsn_3 = np.fromfile('recon/recon_lnsn_3.dat').reshape([-1, NX])
print(recon_cnn_2.shape)
error_cnn_2 = np.sum((data2-recon_cnn_2)**2,axis=1)/np.sum(data2**2,axis=1)
error_cnn_3 = np.sum((data3-recon_cnn_3)**2,axis=1)/np.sum(data3**2,axis=1)
error_lnsn_2 = np.sum((data2-recon_lnsn_2)**2,axis=1)/np.sum(data2**2,axis=1)
error_lnsn_3 = np.sum((data3-recon_lnsn_3)**2,axis=1)/np.sum(data3**2,axis=1)
print((error_cnn_2[-1]+error_cnn_3[-1])/2)
print((error_lnsn_2[-1]+error_lnsn_3[-1])/2)

x = np.linspace(-1,1,128)
t = np.linspace(0,10,1000)

c1 = 'black'
c2 = plt.cm.tab10(0)
plt.figure(figsize=(10,6))
plt.subplot(231)
plt.plot(x,data2[:1001:200].T,c=c1,linewidth=2)
plt.plot(x,recon_lnsn_2[:1001:200].T,'--',c=c2,linewidth=2)
plt.xticks(fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)
plt.title('LNSN',fontsize=fontsize)
plt.xlim(-1,1)
plt.xlabel('$x$',fontsize=fontsize)
plt.ylabel('$u$',fontsize=fontsize)

plt.subplot(232)
plt.plot(x,data2[:1001:200].T,c=c1,linewidth=2)
plt.plot(x,recon_cnn_2[:1001:200].T,'--',c=c2,linewidth=2)
plt.xticks(fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)
plt.title('CNN',fontsize=fontsize)
plt.xlim(-1,1)
plt.xlabel('$x$',fontsize=fontsize)
plt.ylabel('$u$',fontsize=fontsize)

plt.subplot(233)
plt.plot(t, error_cnn_2[:1000],linewidth=2,label='CNN')
plt.plot(t, error_lnsn_2[:1000],linewidth=2,label='LNSN')
plt.xticks(fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)
plt.legend()
plt.title('Error',fontsize=fontsize)
plt.xlim(0,10)
plt.xlabel('$t$',fontsize=fontsize)
plt.ylabel('error',fontsize=fontsize)

plt.subplot(234)
plt.plot(x,data3[:1001:200].T,c=c1,linewidth=2)
plt.plot(x,recon_lnsn_3[:1001:200].T,'--',c=c2,linewidth=2)
plt.xticks(fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)
plt.title('LNSN',fontsize=fontsize)
plt.xlim(-1,1)
plt.xlabel('$x$',fontsize=fontsize)
plt.ylabel('$u$',fontsize=fontsize)

plt.subplot(235)
plt.plot(x,data3[:1001:200].T,c=c1,linewidth=2)
plt.plot(x,recon_cnn_3[:1001:200].T,'--',c=c2,linewidth=2)
plt.xticks(fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)
plt.title('CNN',fontsize=fontsize)
plt.xlim(-1,1)
plt.xlabel('$x$',fontsize=fontsize)
plt.ylabel('$u$',fontsize=fontsize)

plt.subplot(236)
plt.plot(t, error_cnn_3[:1000],linewidth=2,label='CNN')
plt.plot(t, error_lnsn_3[:1000],linewidth=2,label='LNSN')
plt.xticks(fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)
plt.legend()
plt.title('Error',fontsize=fontsize)
plt.xlim(0,10)
plt.xlabel('$t$',fontsize=fontsize)
plt.ylabel('error',fontsize=fontsize)

plt.tight_layout()
plt.show()