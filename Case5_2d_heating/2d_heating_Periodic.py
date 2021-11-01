import numpy as np
import numpy.fft as fft


# Constant parameters
Re = 10
nu = 1/Re
NX,NY = 128,128
LX,LY = 2*np.pi,2*np.pi
deltaX = LX/(NX-1)
deltaY = LY/(NY-1)
deltaT = 0.001
iter_step = 3000


def pre():
    kx1 = np.linspace(-NX / 2, NX / 2, NX, endpoint=False).reshape(NX, 1)
    kx2 = np.linspace(-NX / 2, NX / 2, NX, endpoint=False).reshape(NX, 1) * 1j
    ky1 = kx1.T
    ky2 = kx2.T
    return kx1,kx2,ky1,ky2


def initial():
    u, v = np.zeros([128, 128]), np.zeros([128, 128])
    for i in range(NX):
        for j in range(NY):
            for w in range(9):
                for l in range(9):
                    u[i, j] += np.random.randn() * np.cos((w - 4) * i * deltaX + (l - 4) * i * deltaX) + \
                               np.random.randn() * np.sin((w - 4) * i * deltaX + (l - 4) * i * deltaX)
                    v[i, j] += np.random.randn() * np.cos((w - 4) * i * deltaX + (l - 4) * i * deltaX) + \
                               np.random.randn() * np.sin((w - 4) * i * deltaX + (l - 4) * i * deltaX)

    u = 2 * np.real(u) / np.max(np.real(u)) + (np.random.random([128, 128]) - 0.5) * 4
    v = 2 * np.real(v) / np.max(np.real(v)) + (np.random.random([128, 128]) - 0.5) * 4

    spectral_u = fft.fftshift(fft.fft2(u)) / (NX * NY)
    spectral_v = fft.fftshift(fft.fft2(v)) / (NX * NY)

    spectral_u_extention, spectral_v_extention = np.zeros((int(NX * 2), int(NY * 2))) + 0j, np.zeros(
        (int(NX * 2), int(NY * 2))) + 0j
    _spectral_u_extention, _spectral_v_extention = np.zeros((int(NX * 2), int(NY * 2))) + 0j, np.zeros(
        (int(NX * 2), int(NY * 2))) + 0j

    for epoch in range(1000):
        spectral_u_extention[int(NX / 2):int(NX * 3 / 2), int(NY / 2):int(NY * 3 / 2)] = spectral_u
        spectral_v_extention[int(NX / 2):int(NX * 3 / 2), int(NY / 2):int(NY * 3 / 2)] = spectral_v

        u_extention = fft.ifft2(fft.ifftshift(spectral_u_extention)) * (NX * NY)
        v_extention = fft.ifft2(fft.ifftshift(spectral_v_extention)) * (NX * NY)

        UV = u_extention * v_extention
        UU = u_extention * u_extention
        VV = v_extention * v_extention
        spectral_uv = fft.fftshift(fft.fft2(UV))[int(NX / 2):int(NX * 3 / 2), int(NY / 2):int(NY * 3 / 2)] / (NX * NY)
        spectral_uu = fft.fftshift(fft.fft2(UU))[int(NX / 2):int(NX * 3 / 2), int(NY / 2):int(NY * 3 / 2)] / (NX * NY)
        spectral_vv = fft.fftshift(fft.fft2(VV))[int(NX / 2):int(NX * 3 / 2), int(NY / 2):int(NY * 3 / 2)] / (NX * NY)

        RHS_x = -kx2 * spectral_uu - ky2 * spectral_uv - 0.05 * (kx1 * kx1 + ky1 * ky1) * spectral_u
        RHS_y = -kx2 * spectral_uv - ky2 * spectral_vv - 0.05 * (kx1 * kx1 + ky1 * ky1) * spectral_v

        spectral_u = spectral_u + RHS_x * 0.001
        spectral_v = spectral_v + RHS_y * 0.001

        u = fft.ifft2(fft.ifftshift(spectral_u)) * (NX * NY)
        v = fft.ifft2(fft.ifftshift(spectral_v)) * (NX * NY)

    u = np.sqrt(1 / np.mean(u * u)) * u
    v = np.sqrt(1 / np.mean(v * v)) * v
    spectral_u = fft.fftshift(fft.fft2(u)) / (NX * NY)
    spectral_v = fft.fftshift(fft.fft2(v)) / (NX * NY)

    return u, spectral_u


for dataset_number in range(0,1001):

    kx1,kx2,ky1,ky2 = pre()

    u, spectral_u = initial()

    # Allocate some variables
    # {
    sum_u = np.zeros([iter_step+1,128,128])
    sum_u[0] = np.real(u)
    # }

    # Iterations for Fourier spectral methods
    #{
    for epoch in range(iter_step):
        RHS_x = -nu*(kx1*kx1+ky1*ky1)*spectral_u
        spectral_u = spectral_u + RHS_x*deltaT
        u = fft.ifft2(fft.ifftshift(spectral_u))*(NX*NY)
        u = np.real(u)
        sum_u[epoch+1] = u
        # print(epoch)
    # }

    # Save dataset
    sum_u.tofile('dataset/2d_headting_sdt_'+str(dataset_number)+'.dat')
    print(dataset_number)
