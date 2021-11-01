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

loss_LNSN = np.genfromtxt('loss_data_LNSN.txt')
loss_FSN = np.genfromtxt('loss_data.txt')

plt.yscale('log')
plt.plot(loss_LNSN[:],label='LNSN')
plt.plot(loss_FSN[:],label='FSN')

plt.xlabel('Epochs',fontsize=fontsize)
plt.ylabel('MSE',fontsize=fontsize)

plt.xlim(0,1000000)

plt.legend()
plt.tight_layout()
plt.show()

