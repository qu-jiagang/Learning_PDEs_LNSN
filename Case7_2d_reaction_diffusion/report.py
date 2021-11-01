import numpy as np
import matplotlib.pyplot as plt
import pathlib
from matplotlib import rcParams


rcParams.update({
    "font.size":12,
    "mathtext.fontset":'stix',
    "font.family":"serif",
})
plt.rc('font',family='Times New Roman')
fontsize_ = 12
fontsize = 16

error_cnn = np.genfromtxt('error-cnn.txt')
error_lnsn = np.genfromtxt('error-lnsn.txt')

print(error_cnn.shape)
print(np.mean(error_cnn[:,-1],axis=0))
print(np.min(error_cnn[:,-1],axis=0))
print(np.max(error_cnn[:,-1],axis=0))
print(np.mean(error_lnsn[:,-1],axis=0))
print(np.min(error_lnsn[:,-1],axis=0))
print(np.max(error_lnsn[:,-1],axis=0))

dt = 0.01
Ts = 301

x1 = np.linspace(dt,(Ts-1)*dt,1000)
y1 = np.max(error_cnn,axis=0)
y2 = np.min(error_cnn,axis=0)
y3 = np.max(error_lnsn,axis=0)
y4 = np.min(error_lnsn,axis=0)


plt.fill_between(x1,y3,y4,where=y3>=y4,facecolor='lightblue')
plt.plot(x1,np.mean(error_lnsn,0),label='LNSN')
plt.plot(x1,np.mean(error_cnn,0) ,label='CNN')
plt.xlim(0,3)
plt.ylim(0,0.04)
plt.legend()
plt.xlabel('$t$',fontsize=fontsize)
plt.ylabel('RME',fontsize=fontsize)

plt.show()
