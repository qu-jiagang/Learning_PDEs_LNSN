import numpy as np
from numpy import pi,sin,cos,e
import matplotlib.pyplot as plt

Nx = 128       # number of grids
T  = 10        # total Time
dt = 0.01      # Time step: small dt=0.01 large dt=0.0025
Ts = int(T/dt) # total steps
mu = 0.01      # parameter
x = np.linspace(0,1,Nx)  # uniform grids
deltax = 1/(Nx-1)

q = np.zeros(3)
q[0] =  mu*dt  /deltax/deltax
q[1] = -mu*dt*2/deltax/deltax + 1
q[2] =  mu*dt  /deltax/deltax
print(q)

# datasets with 3 different initial conditions
dataset_1 = np.zeros([Ts,Nx])
dataset_2 = np.zeros([Ts,Nx])
dataset_3 = np.zeros([Ts,Nx])
for t in range(0,Ts):
    for i in range(1,1000):
        M = (-2*i*pi-4*i*pi*cos(i*pi)+(6-i**2*pi**2)*sin(i*pi))/(i**4*pi**4)
        dataset_1[t] += 2*M*np.sin(i*np.pi*x)*np.exp(-mu*i*i*np.pi*np.pi*t*dt)
        M = 30*(2*e**10*i*pi*(700+11*i**2*pi**2)+2*i*pi*(1300+9*i**2*pi**2)*cos(i*pi)-(-12000+60*i**2*pi**2+i**4*pi**4)*sin(i*pi))/(e**10*(100+i**2*pi**2)**3)
        dataset_2[t] += 2*M*np.sin(i*np.pi*x)*np.exp(-mu*i*i*np.pi*np.pi*t*dt)
    dataset_3[t] = np.sin(np.pi*x)*np.exp(-mu*np.pi**2*t*dt)
    print(np.around(t*dt,4))

dataset_1.tofile('dataset/dataset_1_'+str(dt)+'.dat')
dataset_2.tofile('dataset/dataset_2_'+str(dt)+'.dat')
dataset_3.tofile('dataset/dataset_3_'+str(dt)+'.dat')

# plot results for dt
deltaT = int(2/dt)
plt.figure(figsize=(10,3))
plt.subplot(131)
plt.plot(dataset_1[::deltaT].T)
plt.subplot(132)
plt.plot(dataset_2[::deltaT].T)
plt.subplot(133)
plt.plot(dataset_3[::deltaT].T)
plt.show()
