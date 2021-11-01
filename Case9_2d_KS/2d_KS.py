import numpy as np
import pickle
import matplotlib
import pickle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, rc
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt


def initc(x, y):  # Initial condition

    u0 = np.ones((Mx, My))
    for i in range(Mx):
        for j in range(My):
            X, Y = x[i]*1, y[j]*1
            u0[i, j] = np.sin(X + Y) + np.sin(X) + np.sin(Y)

    return u0


def wavenum(Mx, My):  # Wavenumber in Fourier space

    dk = np.pi / l
    kx = np.hstack((np.arange(0., Mx / 2. + 1.), np.arange(-Mx / 2. + 1., 0.))).T * dk
    ky = np.hstack((np.arange(0., My / 2. + 1.), np.arange(-My / 2. + 1., 0.))).T * dk

    return kx, ky


def weights(x, y):  # Spatial integration weights

    weights = np.zeros((Mx, My))
    nx = len(x)
    ny = len(y)
    dx = np.ones_like(x)
    dy = np.ones_like(y)

    for i in range(nx - 1):
        dx[i] = x[i + 1] - x[i]

    dx = np.delete(dx, [len(x) - 1], None)

    for j in range(ny - 1):
        dy[j] = y[j + 1] - y[j]

    dy = np.delete(dy, [len(y) - 1], None)

    for k in range(nx):
        for l in range(ny):
            if k == 0 and l == 0:
                weights[k, l] = dx[0] * dy[0] / 4.
            elif k == 0 and l == ny - 1:
                weights[k, l] = dx[0] * dy[-1] / 4.
            elif k == nx - 1 and l == 0:
                weights[k, l] = dx[-1] * dy[0] / 4.
            elif k == nx - 1 and l == ny - 1:
                weights[k, l] = dx[-1] * dy[-1] / 4.
            elif k == 0 and 0 < l < ny - 1:
                weights[k, l] = dx[0] * (dy[l - 1] + dy[l]) / 4.
            elif k == nx - 1 and 0 < l < ny - 1:
                weights[k, l] = dx[-1] * (dy[l - 1] + dy[l]) / 4.
            elif 0 < k < nx - 1 and l == 0:
                weights[k, l] = (dx[k - 1] + dx[k]) * dy[0] / 4.
            elif 0 < k < nx - 1 and l == ny - 1:
                weights[k, l] = (dx[k - 1] + dx[k]) * dy[-1] / 4.
            else:
                weights[k, l] = (dx[k - 1] + dx[k]) * (dy[l - 1] + dy[l]) / 4.

    return weights


def kssol(u0):  # Solves Kuramoto-Sivashinsky equation in 2-D Fourier space

    kx, ky = wavenum(Mx, My)
    nkx = len(kx)
    nky = len(ky)

    k2 = np.ones((Mx, My))
    kX = np.ones((Mx, My))
    kY = np.ones((Mx, My))
    for i in range(nkx):
        for j in range(nky):
            KX, KY = kx[i], ky[j]
            k2[i, j] = -(KX ** 2) - ((nu2 / nu1) * (KY ** 2)) + (
                        nu1 * ((KX ** 4) + (2. * (nu2 / nu1) * (KX ** 2) * (KY ** 2)) + ((nu2 / nu1) ** 2 * (KY ** 4))))
            kX[i, j] = KX
            kY[i, j] = KY

    u0spec = np.fft.fft2(u0)  # Initial condition in Fourier space

    nlinspecx = np.zeros((Mx, My, nt + 1), dtype='complex')  # Nonlinear part x
    nlinspecy = np.zeros((Mx, My, nt + 1), dtype='complex')  # Nonlinear part y
    A = np.ones((Mx, My))

    u = np.zeros((Mx, My, nt + 1), dtype='complex')  # Variable in Fourier space
    u[:, :, 0] = u0spec
    ur = np.zeros((Mx, My, nt + 1), dtype='complex')  # Variable in real space
    ur[:, :, 0] = u0
    en = np.zeros((nt + 1))  # Energy calculation
    wt = weights(x, y)
    ur2 = ur[:, :, 0] * ur[:, :, 0]
    en[0] = np.dot(wt.flatten(), ur2.flatten())
    nlin = np.zeros((nt + 1))

    for i in range(nt):
        print(i)
        if i == 0:
            u[:, :, i + 1] = (u[:, :, i] + (dt * (nlinspecx[:, :, i] + nlinspecy[:, :, i] + (c * A * u[:, :, i])))) / (
                        A + (dt * (k2 + (c * A))))
            ur[:, :, i + 1] = np.fft.ifft2(u[:, :, i + 1]).real
            ur[:, :, i + 1] = ur[:, :, i + 1] - (
                        (1. / (4. * (np.pi ** 2))) * np.dot(wt.flatten(), ur[:, :, i + 1].flatten()) * A)
            ur2 = ur[:, :, i + 1] * ur[:, :, i + 1]
            en[i + 1] = np.dot(wt.flatten(), ur2.flatten())
        else:
            u[:, :, i] = np.fft.fft2(ur[:, :, i])
            nlinspecx[:, :, i] = -0.5 * np.fft.fft2(
                np.absolute(np.fft.ifft2(1j * kX * u[:, :, i])) * np.absolute(np.fft.ifft2(1j * kX * u[:, :, i])))
            nlinspecy[:, :, i] = -0.5 * (nu2 / nu1) * np.fft.fft2(
                np.absolute(np.fft.ifft2(1j * kY * u[:, :, i])) * np.absolute(np.fft.ifft2(1j * kY * u[:, :, i])))
            u[:, :, i + 1] = ((4. * u[:, :, i]) - u[:, :, i - 1] + (
                        4. * dt * (nlinspecx[:, :, i] + nlinspecy[:, :, i] + (c * A * u[:, :, i]))) - (2. * dt * (
                        nlinspecx[:, :, i - 1] + nlinspecy[:, :, i - 1] + (c * A * u[:, :, i - 1])))) / (
                                         (3. * A) + (2. * dt * (k2 + (c * np.ones_like(k2)))))
            ur[:, :, i + 1] = np.fft.ifft2(u[:, :, i + 1]).real
            ur[:, :, i + 1] = ur[:, :, i + 1] - (
                        (1. / (4. * (np.pi ** 2))) * np.dot(wt.flatten(), ur[:, :, i + 1].flatten()) * A)
            ur2 = ur[:, :, i + 1] * ur[:, :, i + 1]
            en[i + 1] = np.dot(wt.flatten(), ur2.flatten())

    return ur, en


# Bifurcation parameters
nu1 = 0.1            					# nu1 = (pi/Lx)^2
nu2 = 0.1								# nu2 = (pi/Ly)^2

Lx = np.pi/np.sqrt(nu1)                 # Size of domain in x
Ly = np.pi/np.sqrt(nu2)	              # Size of domain in y

# Number of modes
Mx = 64  # Number of modes in x
My = 64  # Number of modes in y

c = 100.

# Run time 
Tf = 100      # Final time
nt = 40000   # Number of time steps

# Step-size
dt = Tf / nt  # Size of the time step

t = np.linspace(0., Tf, nt + 1)

# Cell size
l = np.pi
dx = (2. * l) / (Mx)  # Grid spacing in x
dy = (2. * l) / (My)  # Grid spacing in y

# Grid  
x = np.arange(0., Mx) * dx  # Grid points in x
y = np.arange(0., My) * dy  # Grid points in y
X, Y = np.meshgrid(x, y, indexing='ij')  # Meshgrid in x-y

u0 = initc(x, y)

ur, en = kssol(u0)

ur = np.real(ur)
ur = np.transpose(ur,[2,0,1])
ur.tofile('2d-KS_dataset.dat')
print(ur.shape)

plt.figure()
plt.plot(ur[::10,7::16,16])

plt.figure(figsize=(10,2))
for i in range(10):
    plt.subplot2grid((1,10),(0,i))
    plt.imshow(ur[i*int(nt/10)],vmin=np.mean(ur),vmax=np.max(ur))
plt.show()
