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


error_lnsn = np.zeros([12,1000,301])
i = -1
for layer in range(1,4):
    for node in [8,16,32,64]:
        i += 1
        error_lnsn[i] = np.genfromtxt('error/error-lnsn-'+str(layer)+'-'+str(node)+'.txt')

        print('layer = ', layer, '; nodes = ', node, ';')
        print('max error = ', np.max(error_lnsn[i,:,-1]))
        print('min error = ', np.min(error_lnsn[i,:,-1]))
        print('mean error = ', np.mean(error_lnsn[i,:,-1]))
        print('------')
