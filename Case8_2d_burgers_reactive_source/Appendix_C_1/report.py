import numpy as np
import matplotlib.pyplot as plt
import pathlib
from matplotlib import rcParams


rcParams.update({
    "font.size":18,
    "mathtext.fontset":'stix',
    "font.family":"serif",
})
plt.rc('font',family='Times New Roman')
fontsize_ = 18
fontsize = 22

error_1 = np.genfromtxt('error-lnsn-1.txt')
error_2 = np.genfromtxt('error-lnsn-2.txt')

print(error_1.shape)

dt = 0.01
Ts = 301

x1 = np.linspace(0,(Ts-1)*dt,301)
y1 = np.max(error_1,axis=0)
y2 = np.min(error_1,axis=0)
y3 = np.max(error_2,axis=0)
y4 = np.min(error_2,axis=0)
print(error_2.shape)
print(np.mean(error_1[:,-1],axis=0))
print(np.min(error_1[:,-1],axis=0))
print(np.max(error_1[:,-1],axis=0))
print(np.mean(error_2[:,-1],axis=0))
print(np.min(error_2[:,-1],axis=0))
print(np.max(error_2[:,-1],axis=0))

plt.fill_between(x1,y3,y4,where=y3>=y4,facecolor='lightblue')
plt.plot(x1,np.mean(error_2,0),label='$x_L+x_N$')
plt.plot(x1,np.mean(error_1,0) ,label='$x_L$')
plt.xlim(0,3)
plt.ylim(0,1)
plt.legend()
plt.xlabel('$t$',fontsize=fontsize)
plt.ylabel('RME',fontsize=fontsize)

plt.tight_layout()
plt.show()
