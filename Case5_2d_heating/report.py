import numpy as np
import matplotlib.pyplot as plt
import pathlib
from matplotlib import rcParams


rcParams.update({
    "font.size":14,
    "mathtext.fontset":'stix',
    "font.family":"serif",
})
plt.rc('font',family='Times New Roman')
fontsize_ = 12
fontsize = 16


NX = NY = 128
Ts1 = 3001
Ts2 = 301
dt1 = 0.001
dt2 = 0.01

error = np.genfromtxt('error.txt')
error_ldt = np.genfromtxt('error-ldt.txt')
print(error_ldt.shape)
print(np.mean(error_ldt[:,-1],axis=0))
print(np.min(error_ldt[:,-1],axis=0))
print(np.max(error_ldt[:,-1],axis=0))


x1 = np.linspace(dt1,(Ts1-1)*dt1,np.size(error,1))
y1 = np.max(error,axis=0)
y2 = np.min(error,axis=0)
x2 = np.linspace(0,(Ts2-1)*dt2,np.size(error_ldt,1))
y3 = np.max(error_ldt,axis=0)
y4 = np.min(error_ldt,axis=0)
ymin = np.min(error_ldt)
ymax = np.max(error_ldt)

ax1 = plt.subplot(111)
ax1.set_xlim(0,3)
ax1.set_ylim(ymin,ymax)

ax1.fill_between(x1,y1,y2,where=y1>=y2,facecolor='lightblue',alpha=0.7)
ax1.plot(x1,np.mean(error,axis=0),c=plt.cm.tab10(0),label='$\delta t = 0.001$')
ax1.fill_between(x2,y3,y4,where=y3>=y4,facecolor='wheat',alpha=0.7)
ax1.plot(x2,np.mean(error_ldt,axis=0),c=plt.cm.tab10(1),label='$\delta t = 0.01$')

ax1.set_xlabel('$t$',fontsize=fontsize)
ax1.set_ylabel('RME',fontsize=fontsize)
plt.xticks(fontsize=fontsize_)
plt.yticks(fontsize=fontsize_)
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))

plt.legend()
plt.tight_layout()
plt.show()
