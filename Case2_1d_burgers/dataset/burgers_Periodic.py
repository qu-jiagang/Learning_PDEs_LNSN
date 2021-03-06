# Fourier spectrum method for 1d burgers equation with Periodic boundary conditions

import numpy
import matplotlib.pyplot as plt
import math
from numpy import fft
import symbol
import numpy as np
from math import *
from scipy import interpolate


def RHS(spectral_u, K, on):
    coef = [0.0, 0.5, 0.5, 1.0]
    spectral_u = spectral_u + coef[on] * ds * K
    spectral_u_extend[:int(NX / 2)] = 0 + 0j
    spectral_u_extend[int(NX * 3 / 2):] = 0 + 0j
    spectral_u_extend[int(NX / 2):int(NX * 3 / 2)] = spectral_u
    u_extention = fft.ifft(fft.ifftshift(spectral_u_extend))
    w_extention = u_extention * u_extention
    spectral_w = fft.fftshift(fft.fft(w_extention))
    rhs = -complex_k * spectral_w[int(NX / 2):int(NX * 3 / 2)] - k * k * nu * spectral_u
    return rhs


Re = 100
nu = 1./Re
NX = 256
print(NX)
LX = 2
deltaX = LX/(NX-1)
ds = 0.001        # time step;
Ts = 10001

k = numpy.linspace(-NX/2,NX/2,NX,endpoint=False)/(LX/(2*pi))
complex_k = k*1j

x = numpy.linspace(-1,1,NX,endpoint=False)

for no in range(3):
    print(no)

    # 3 different initial conditions
    u_initial = np.zeros([3,NX])
    u_initial[0] = -np.sin(pi*x)
    u_initial[2] = -10*x*(1+x)*(1-x)/(1+np.exp(10*x**2))
    u_initial[4] = -2*x/(1+np.sqrt(1/np.exp(100/8))*np.exp(100*x**2/4))

    # initial condition
    u = u_initial[no]

    spectral_u = fft.fftshift(fft.fft(u))
    spectral_u_extend = numpy.zeros(int(NX*2))+0j

    sum_u = np.zeros([Ts,NX])
    sum_u[0] = u
    K0 = numpy.zeros(NX)+0j
    for step in range(Ts-1):
        K1 = RHS(spectral_u, K0 ,0)
        K2 = RHS(spectral_u, K1, 1)
        K3 = RHS(spectral_u, K2, 2)
        K4 = RHS(spectral_u, K3, 3)
        spectral_u = spectral_u + ds/6.0*(K1+2.0*K2+2.0*K3+K4)
        u = fft.ifft(fft.ifftshift(spectral_u))

        sum_u[step+1] = np.real(u)

    # x = x.reshape(NX,)
    data_line = np.zeros((Ts,128))
    xx = np.linspace(-1,1,128,endpoint=False)
    for i in range(Ts):
        f = interpolate.interp1d(x,sum_u[i],kind='quadratic')
        data_line[i] = f(xx)

    data_line.tofile('1d_burgers_Periodic_'+str(no)+'.dat')
