import numpy as np
import matplotlib.pyplot as plt


Ts = 1000
NX = 402
n = -1
for node in ['8','16','32','64','128']:
    n+=1
    for i in range(1,3):
        no = 2*i
        data  = np.fromfile('../dataset/1d_burgers_Periodic_'+str(no)+'.dat').reshape((10001,NX))[::10][1:]
        recon_lnsn = np.fromfile('reconst/recon_CNN_nodes'+node+'_'+str(no)+'.dat').reshape((Ts,NX))
        recon_cnn  = np.fromfile('reconst/recon_LNSN_nodes'+node+'_'+str(no)+'.dat').reshape((Ts,NX))

        error_lnsn = np.sum((data-recon_lnsn)**2,axis=1)/np.sum(data**2,axis=1)
        error_cnn  = np.sum((data-recon_cnn )**2,axis=1)/np.sum(data**2,axis=1)

        plt.plot(error_lnsn,c=plt.cm.tab10(n))
        plt.plot(error_cnn,'--',c=plt.cm.tab10(n))
        print(node, no)

plt.show()