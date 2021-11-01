import numpy as np
import matplotlib.pyplot as plt
from math import *

NX = 128
NY = 128
Nt = 1100
dt = 1.0e-3
Lx = 2
Ly = 2
dx = Lx/(NX-1)
dy = Ly/(NY-1)
Re = 100
eps = 0.1

print(1-4*dt/dx/dx/Re)

def RK4(u):
    RHS = np.zeros([NX,NY])
    duudxx = np.zeros([NX,NY])
    duudxx[1:-1,1:-1] = (u[2:,1:-1]-2*u[1:-1,1:-1]+u[:-2,1:-1])/dx/dx \
                      + (u[1:-1,2:]-2*u[1:-1,1:-1]+u[1:-1,:-2])/dy/dy
    RHS[1:-1,1:-1] = 1/Re*duudxx[1:-1,1:-1] + (u[1:-1,1:-1]-u[1:-1,1:-1]**3)/eps/eps
    return RHS


def boundary(u):
    u[   0,  :] = u[   1,  :]
    u[  -1,  :] = u[  -2,  :]
    u[1:-1,  0] = u[1:-1,  1]
    u[1:-1, -1] = u[1:-1, -2]
    u[ 0,  0] = 0.5 * (u[ 1,  0] + u[ 0,  1])
    u[ 0, -1] = 0.5 * (u[ 0, -2] + u[ 1, -1])
    u[-1,  0] = 0.5 * (u[-1,  1] + u[-2,  0])
    u[-1, -1] = 0.5 * (u[-1, -2] + u[-2, -1])
    return u

for no in range(1,1001):
    print(no)

    # initial condition
    u = (np.random.rand(NX,NY)*2-1)*0.1

    n = -1
    u = boundary(u)
    sum_u = np.zeros([Nt, NX, NY])
    for step in range(Nt):
        K1 = RK4(u)
        K2 = RK4(u+dt/2*K1)
        K3 = RK4(u+dt/2*K2)
        K4 = RK4(u+dt*K3)
        u = u + dt/6*(K1+2*K2+2*K3+K4)
        u = boundary(u)

        if step%1==0:
            n += 1
            sum_u[n] = np.real(u)

    sum_u.tofile('dataset/2d-Allen-Cahn-'+str(no)+'.dat')

    # plot
    # for i in range(4):
    #     plt.subplot(221+i)
    #     plt.imshow(sum_u[i*int((Nt-110)/3)+100])
    # plt.show()
