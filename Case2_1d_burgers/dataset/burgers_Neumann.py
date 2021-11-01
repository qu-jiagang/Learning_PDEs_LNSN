import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
from math import *
from chebpy import cheb_D1_mat
from chebpy import etdrk4_coeff_ndiag
from scipy.linalg import expm, inv
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy import interpolate


for no in range(3):
    Re = 100
    Lx = 2
    ds = 0.001     # time step;
    N = 256-1
    Ts = 10001
    deltax = Lx / N
    q = np.zeros(3)
    q[0] =  1/Re*  ds/deltax/deltax
    q[1] = -1/Re*2*ds/deltax/deltax+1
    q[2] =  1/Re*  ds/deltax/deltax
    print(q)

    print(no)

    D, x = cheb_D1_mat(N)
    x = np.flip(x)

    # 3 different initial conditions
    u_initial = np.zeros([3,N+1,1])
    u_initial[0] = 10*x*(1+x)*(1-x)/(1+np.exp(10*x**2))
    u_initial[1] = 2*x/(1+np.sqrt(1/np.exp(100/8))*np.exp(100*x**2/4))
    u_initial[2] = np.sin(0.5*pi*x)

    # initial condition
    u = u_initial[no]

    v = u.copy()

    W = D

    h = ds
    M = 32
    R = 15.
    D1 = np.zeros_like(D)
    D1[1:N, :] = D[1:N, :]
    L = np.dot(D, D1)/Re
    L = (4. / Lx ** 2) * L
    Q, f1, f2, f3 = etdrk4_coeff_ndiag(L, h, M, R)

    A = h*L
    E = expm(A)
    E2 = expm(A/2)

    sum_u = np.zeros((Ts,N+1))

    t = 0
    sum_u[0] = u.reshape(N+1)
    for j in range(Ts-1):
        Nu = v*np.dot(W,v)
        a = np.dot(E2, v) + np.dot(Q, Nu)
        Na = a*np.dot(W,a)
        b = np.dot(E2, v) + np.dot(Q, Na)
        Nb = b*np.dot(W,b)
        c = np.dot(E2, a) + np.dot(Q, 2*Nb-Nu)
        Nc = c*np.dot(W,c)
        v = np.dot(E, v) + np.dot(f1, Nu) + 2*np.dot(f2, Na+Nb) + np.dot(f3, Nc)
        t = ds*j

        sum_u[j+1] = v.reshape((N+1))

    x = x.reshape(N+1,)
    data_line = np.zeros((Ts,128))
    xx = np.linspace(-1,1,128)
    for i in range(Ts):
        f = interpolate.interp1d(x,sum_u[i],kind='quadratic')
        data_line[i] = f(xx)

    data_line.tofile('1d_burgers_Neumann_'+str(no)+'.dat')

