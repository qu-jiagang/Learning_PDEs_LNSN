import numpy as np
import matplotlib.pyplot as plt


Ts = 10001
NX = 402

data = np.zeros([12,1001,NX])
CNN = np.zeros([12,1000,NX])
LNSN = np.zeros([12,1000,NX])
i = -1

for no in range(3):
    i += 1
    data[i] = np.fromfile('../dataset/1d_burgers_Periodic_'+str(no)+'.dat').reshape([Ts,NX])[:10001:10]
    CNN[i]  = np.fromfile('reconst/recon_CNN_' +str(no)+'.dat').reshape([1000,NX])
    LNSN[i] = np.fromfile('reconst/recon_LNSN_'+str(no)+'.dat').reshape([1000,NX])

error1 = np.sum((data[:,1:]-CNN )**2,axis=2)/np.sum((data[:,1:])**2,axis=2)
error2 = np.sum((data[:,1:]-LNSN)**2,axis=2)/np.sum((data[:,1:])**2,axis=2)

for i in range(0,3):
    i = i*2
    plt.plot(error1[i])
for i in range(0,3):
    i = i*2
    plt.plot(error2[i],'--')
plt.show()

plt.figure()
for i in range(3):
    i = i
    plt.subplot2grid((2, 3), (0, i))
    plt.plot(data[i, ::100, :].T)
    plt.plot(LNSN[i,1::100, :].T, '--')
    plt.subplot2grid((2, 3), (1, i))
    plt.plot(data[i, ::100, :].T)
    plt.plot(CNN[i, 1::100, :].T, '--')
    plt.ylabel(i,loc='center')

plt.show()