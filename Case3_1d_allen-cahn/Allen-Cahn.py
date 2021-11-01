import numpy as np
from chebpy import cheb_D1_mat
from scipy.linalg import expm, inv
import matplotlib.pyplot as plt
from scipy import interpolate
from numpy import *


'''
Solution of Allen-Cahn equation by ETDRK4 shceme.

u_t = eps*u_xx + u - u^3 on
    [-1,1], u(-1) = -1, u(1) = 1

Computation is based on Chebyshev points, so linear term is
non-diagonal.
'''

N = 128
D, xx = cheb_D1_mat(N)
x = xx[1:N]
# w = .53*x + .47*np.sin(-1.5*np.pi*x) - x
# w = .4*x + .6*np.sin(-1.5*np.pi*x) - x
# w = .62*x + .38*np.sin(-1.5*np.pi*x) - x
w = .68*x + .32*np.sin(-1.5*np.pi*x) - x

u = np.concatenate(([[1]], w+x, [[-1]]))

h = 1./100
M = 32 # Number of points in upper half-plane
kk = np.arange(1, M+1)
r = 15.0 * np.exp(1j * np.pi * (kk - .5) / M)
L = np.dot(D, D) # L = D^2
eps = 0.01
L = eps * L[1:N,1:N]
A = h * L
E = expm(A)
E2 = expm(A/2)
I = np.eye(N-1)
Z = 1j * np.zeros((N-1,N-1))
f1 = Z; f2 = Z; f3 = Z; Q = Z
for j in range(M):
    z = r[j]
    zIA = inv(z * I - A)
    hzIA = h * zIA
    hzIAz2 = hzIA / z**2
    Q = Q + hzIA * (np.exp(z/2) - 1)
    f1 = f1 + hzIAz2 * (-4 - z + np.exp(z) * (4 - 3*z + z**2))
    f2 = f2 + hzIAz2 * (2 + z + np.exp(z) * (z - 2))
    f3 = f3 + hzIAz2 * (-4 - 3*z - z*z + np.exp(z) * (4 - z))
f1 = np.real(f1 / M)
f2 = np.real(f2 / M)
f3 = np.real(f3 / M)
Q = np.real(Q / M)

tt = 0.
tmax = 10
nmax = int(round(tmax / h))
nplt = 1

sum_u = np.zeros([nmax,N+1])

for n in range(nmax):
    t = (n+1) * h
    Nu = (w+x) - np.power(w+x, 3)
    a = np.dot(E2, w) + np.dot(Q, Nu)
    Na = (a+x) - np.power(a+x, 3)
    b = np.dot(E2, w) + np.dot(Q, Na)
    Nb = (b+x) - np.power(b+x, 3)
    c = np.dot(E2, a) + np.dot(Q, 2*Nb-Nu)
    Nc = (c+x) - np.power(c+x, 3)
    w = np.dot(E, w) + np.dot(f1, Nu) + 2 * np.dot(f2, Na+Nb) + \
        np.dot(f3, Nc)
    if ((n+1) % nplt) == 0:
        print (n+1)
        u = np.concatenate(([[1]],w+x,[[-1]])).T
        sum_u[n] = u


data_line = np.zeros((nmax,128))
x = np.linspace(-1,1,128)
for i in range(nmax):
    f = interpolate.interp1d(xx.flatten(),sum_u[i],kind='quadratic')
    data_line[i] = f(x)


# data_line.tofile('Allen-Cahn_3.dat')

x, y = np.meshgrid(np.linspace(-1,1,128), np.linspace(0,tmax,nmax))
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_wireframe(x,y,data_line)
plt.show()


