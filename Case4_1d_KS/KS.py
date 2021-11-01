import numpy as np
import pickle
import matplotlib.pyplot as plt
from math import pi as PI
from decimal import Decimal
from matplotlib import cm, rc
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import pyfftw


def initc(x):  # Initial condition
    u0 = np.random.rand(np.size(x))*0.25
    return u0


def wavenum(Mx):  # Wavenumber evaluation in Fourier space

    kxx = np.fft.rfftfreq(Mx, 1. / Mx).astype(float)
    kx = np.zeros_like(kxx, dtype='longdouble')
    kx = kxx / L

    return kx


def fftrtf(uspec):  # Replicate half of the variable for symmetry in Fourier space

    rtf = pyfftw.empty_aligned((Mx,), dtype='clongdouble')
    usp = np.conjugate(uspec[::-1])
    uspect = np.delete(usp, [0, Mx // 2], None)
    rtf = np.concatenate((uspec[:], uspect[:]), axis=0)

    return rtf


def weights(x):  # Spatial integration weights

    weights = np.empty_like(x, dtype='longdouble')
    dx = np.empty_like(x, dtype='longdouble')
    nx = len(x)
    for i in range(nx - 1):
        dx[i] = x[i + 1] - x[i]

    dx = np.delete(dx, [len(x) - 1], None)

    for j in range(nx):
        if j == 0:
            weights[j] = dx[0] / 2
        elif j == nx - 1:
            weights[j] = dx[-1] / 2
        else:
            weights[j] = dx[j - 1] / 2 + dx[j] / 2

    return weights


def antialias(uhat, vhat):  # Anti-aliasing using padding technique

    N = len(uhat)
    M = 2 * N
    uhat_pad = np.concatenate((uhat[0:N / 2], np.zeros((M - N,)), uhat[N / 2:]), axis=0)
    vhat_pad = np.concatenate((vhat[0:N / 2], np.zeros((M - N,)), vhat[N / 2:]), axis=0)
    u_pad = pyfftw.interfaces.numpy_fft.ifft(uhat_pad)
    v_pad = pyfftw.interfaces.numpy_fft.ifft(vhat_pad)
    w_pad = u_pad * v_pad
    what_pad = pyfftw.interfaces.numpy_fft.fft(w_pad)
    what = 2. * np.concatenate((what_pad[0:N / 2], what_pad[M - N / 2:M]), axis=0)

    return what


def aalias(uhat):  # To calculate (uhat)^2 in real space and transform to Fourier space

    ureal = pyfftw.interfaces.numpy_fft.irfft(uhat)
    nlt = ureal.real * ureal.real
    usp = pyfftw.interfaces.numpy_fft.rfft(nlt)

    return usp


def alias(uht):  # To calculate (uht)^2 in real space

    url = pyfftw.interfaces.numpy_fft.ifft(uht)
    nlter = url.real * url.real

    return nlter


def fwnum(Mx):
    alpha = np.fft.fftfreq(Mx, 1. / Mx).astype(int)
    alpha[Mx // 2] *= -1

    return alpha


def kssol1(u0):  # Solver for start at time t = 0

    Tf = Decimal("300.0")  # Final time
    t = Decimal("0.0")  # Current time
    h = Decimal("0.025")  #0.25
    dt = float(h)  # Size of the time step
    nt = int(Tf / h)

    kx = wavenum(Mx)

    A = np.ones((Mx // 2 + 1,))
    k2 = -(kx ** 2) + (kx ** 4)

    u = pyfftw.empty_aligned((Mx // 2 + 1, nt + 1), dtype='clongdouble')
    us0 = pyfftw.empty_aligned((Mx // 2 + 1,), dtype='clongdouble')
    u[:, 0] = pyfftw.interfaces.numpy_fft.rfft(u0)
    u[0, 0] -= u[0, 0]
    nlin = pyfftw.empty_aligned((Mx // 2 + 1, nt + 1), dtype='clongdouble')
    nlin[:, 0] = aalias(u[:, 0])
    nlinspec = pyfftw.empty_aligned((Mx // 2 + 1, nt + 1), dtype='clongdouble')
    nlinspec[:, 0] = -0.5 * 1j * kx * nlin[:, 0]
    nls = pyfftw.empty_aligned((Mx // 2 + 1,), dtype='clongdouble')
    nlspec = pyfftw.empty_aligned((Mx // 2 + 1,), dtype='clongdouble')
    nondx = pyfftw.empty_aligned((Mx, nt + 1), dtype='longdouble')
    nondx2 = pyfftw.empty_aligned((Mx, nt + 1), dtype='longdouble')
    nondx[:, 0] = alias(1j * fwnum(Mx) * fftrtf(u[:, 0]))
    nondx2[:, 0] = alias(-1. * ((fwnum(Mx)) ** 2) * fftrtf(u[:, 0]))
    ur = pyfftw.empty_aligned((Mx, nt + 1), dtype='longdouble')
    ur[:, 0] = pyfftw.interfaces.numpy_fft.irfft(u[:, 0]).real
    wt = weights(x)
    en = np.empty((nt + 1,), dtype='longdouble')
    en[0] = np.dot(wt, ur[:, 0] * ur[:, 0])
    ent = np.empty((nt + 1,), dtype='longdouble')
    ent[0] = (2. * np.dot(wt, nondx[:, 0])) - (2. * np.dot(wt, nondx2[:, 0]))

    for i in range(nt):
        t += h
        print(i)
        if i == 0:
            us0 = (u[:, i] + (dt * nlinspec[:, i])) / (A + (dt * k2))
            us0[0] -= us0[0]
            us0[-1] -= us0[-1]
            nls[:] = aalias(0.5 * (u[:, 0] + us0))
            nlspec[:] = -0.5 * 1j * kx * nls[:]
            u[:, i + 1] = (u[:, i] - (0.5 * dt * k2 * u[:, i]) + (dt * nlspec[:])) / (A + (0.5 * dt * k2))
            u[0, i + 1] -= u[0, i + 1]
            u[-1, i + 1] -= u[-1, i + 1]
            ur[:, i + 1] = pyfftw.interfaces.numpy_fft.irfft(u[:, i + 1]).real
            en[i + 1] = np.dot(wt, ur[:, i + 1] * ur[:, i + 1])
            nondx[:, i + 1] = alias(1j * fwnum(Mx) * fftrtf(u[:, i + 1]))
            nondx2[:, i + 1] = alias(-1. * ((fwnum(Mx)) ** 2) * fftrtf(u[:, i + 1]))
            ent[i + 1] = (2. * np.dot(wt, nondx[:, i + 1])) - (2. * np.dot(wt, nondx2[:, i + 1]))
        elif i == 1:
            nlin[:, i] = aalias(u[:, i])
            nlinspec[:, i] = -0.5 * 1j * kx * nlin[:, i]
            u[:, i + 1] = ((4 * u[:, i]) - u[:, i - 1] + (4 * dt * nlinspec[:, i]) - (2 * dt * nlinspec[:, i - 1])) / (
                        (3 * A) + (2 * dt * k2))
            u[0, i + 1] -= u[0, i + 1]
            u[-1, i + 1] -= u[-1, i + 1]
            ur[:, i + 1] = pyfftw.interfaces.numpy_fft.irfft(u[:, i + 1]).real
            en[i + 1] = np.dot(wt, ur[:, i + 1] * ur[:, i + 1])
            nondx[:, i + 1] = alias(1j * fwnum(Mx) * fftrtf(u[:, i + 1]))
            nondx2[:, i + 1] = alias(-1. * ((fwnum(Mx)) ** 2) * fftrtf(u[:, i + 1]))
            ent[i + 1] = (2. * np.dot(wt, nondx[:, i + 1])) - (2. * np.dot(wt, nondx2[:, i + 1]))
        else:
            nlin[:, i] = aalias(u[:, i])
            nlinspec[:, i] = -0.5 * 1j * kx * nlin[:, i]
            u[:, i + 1] = ((18 * u[:, i]) - (9 * u[:, i - 1]) + (2 * u[:, i - 2]) + (18 * dt * nlinspec[:, i]) - (
                        18 * dt * nlinspec[:, i - 1]) + (6 * dt * nlinspec[:, i - 2])) / ((11 * A) + (6 * dt * k2))
            u[0, i + 1] -= u[0, i + 1]
            u[-1, i + 1] -= u[-1, i + 1]
            ur[:, i + 1] = pyfftw.interfaces.numpy_fft.irfft(u[:, i + 1]).real
            en[i + 1] = np.dot(wt, ur[:, i + 1] * ur[:, i + 1])
            nondx[:, i + 1] = alias(1j * fwnum(Mx) * fftrtf(u[:, i + 1]))
            nondx2[:, i + 1] = alias(-1. * ((fwnum(Mx)) ** 2) * fftrtf(u[:, i + 1]))
            ent[i + 1] = (2. * np.dot(wt, nondx[:, i + 1])) - (2. * np.dot(wt, nondx2[:, i + 1]))

    return u, ur, en, ent


Mx = 2 ** 6  # Number of modes
L = 22/PI/2   # Length of domain
dx = (2. * L * PI) / np.float(Mx)
x = np.arange(0., Mx) * dx

u0 = initc(x)  # Initial condition

u, ur, en, ent = kssol1(u0)

data = np.transpose(ur,(1,0))
data.tofile('1d_ks_L22.dat')

# Plot contour map of the solution
plt.imshow(data.T[:,::20])
plt.show()