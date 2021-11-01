import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft


# Constant parameters
nu = 0.015
NX,NY = 128,128
LX,LY = 2*np.pi,2*np.pi
deltaX = LX/(NX-1)
deltaY = LY/(NY-1)
deltaT = 0.001
iter_step = 3001

for no in range(547,1001):
    # Initial condition
    # {
    kx1 = np.linspace(-NX/2,NX/2,NX,endpoint=False).reshape(NX,1)
    kx2 = np.linspace(-NX/2,NX/2,NX,endpoint=False).reshape(NX,1)*1j
    ky1 = kx1.T
    ky2 = kx2.T

    u,v = np.zeros([128,128]),np.zeros([128,128])
    for i in range(NX):
        for j in range(NY):
            for w in range(9):
                for l in range(9):
                    u[i,j] += np.random.randn()*np.cos((w-4)*i*deltaX+(l-4)*i*deltaX) +\
                              np.random.randn()*np.sin((w-4)*i*deltaX+(l-4)*i*deltaX)
                    v[i,j] += np.random.randn()*np.cos((w-4)*i*deltaX+(l-4)*i*deltaX) +\
                              np.random.randn()*np.sin((w-4)*i*deltaX+(l-4)*i*deltaX)

    u = 2*np.real(u)/np.max(np.real(u))+(np.random.random([128,128])-0.5)*4
    v = 2*np.real(v)/np.max(np.real(v))+(np.random.random([128,128])-0.5)*4

    spectral_u = fft.fftshift(fft.fft2(u))/(NX*NY)
    spectral_v = fft.fftshift(fft.fft2(v))/(NX*NY)

    spectral_u_extention, spectral_v_extention = np.zeros((int(NX*2),int(NY*2)))+0j, np.zeros((int(NX*2),int(NY*2)))+0j
    _spectral_u_extention, _spectral_v_extention = np.zeros((int(NX*2),int(NY*2)))+0j, np.zeros((int(NX*2),int(NY*2)))+0j

    for epoch in range(1000):
        spectral_u_extention[int(NX/2):int(NX*3/2),int(NY/2):int(NY*3/2)] = spectral_u
        spectral_v_extention[int(NX/2):int(NX*3/2),int(NY/2):int(NY*3/2)] = spectral_v

        u_extention = fft.ifft2(fft.ifftshift(spectral_u_extention))*(NX*NY)
        v_extention = fft.ifft2(fft.ifftshift(spectral_v_extention))*(NX*NY)

        UV = u_extention*v_extention
        UU = u_extention*u_extention
        VV = v_extention*v_extention
        spectral_uv = fft.fftshift(fft.fft2(UV))[int(NX/2):int(NX*3/2),int(NY/2):int(NY*3/2)]/(NX*NY)
        spectral_uu = fft.fftshift(fft.fft2(UU))[int(NX/2):int(NX*3/2),int(NY/2):int(NY*3/2)]/(NX*NY)
        spectral_vv = fft.fftshift(fft.fft2(VV))[int(NX/2):int(NX*3/2),int(NY/2):int(NY*3/2)]/(NX*NY)

        RHS_x = -kx2*spectral_uu-ky2*spectral_uv-0.05*(kx1*kx1+ky1*ky1)*spectral_u
        RHS_y = -kx2*spectral_uv-ky2*spectral_vv-0.05*(kx1*kx1+ky1*ky1)*spectral_v

        spectral_u = spectral_u + RHS_x*0.001
        spectral_v = spectral_v + RHS_y*0.001

        u = fft.ifft2(fft.ifftshift(spectral_u))*(NX*NY)
        v = fft.ifft2(fft.ifftshift(spectral_v))*(NX*NY)

    u = np.sqrt(1/np.mean(u*u))*u
    v = np.sqrt(1/np.mean(v*v))*v

    spectral_u = fft.fftshift(fft.fft2(u)) / (NX * NY)
    spectral_v = fft.fftshift(fft.fft2(v)) / (NX * NY)
    # }

    # Allocate some variables
    # {
    energy = np.zeros(iter_step)
    sum_u = np.zeros([iter_step,128,128])
    sum_v = np.zeros([iter_step,128,128])
    # }

    # Iterations for Fourier spectral methods
    #{
    for epoch in range(iter_step):
        spectral_u_extention[int(NX/2):int(NX*3/2),int(NY/2):int(NY*3/2)] = spectral_u
        spectral_v_extention[int(NX/2):int(NX*3/2),int(NY/2):int(NY*3/2)] = spectral_v

        u_extention = fft.ifft2(fft.ifftshift(spectral_u_extention))*(NX*NY)
        v_extention = fft.ifft2(fft.ifftshift(spectral_v_extention))*(NX*NY)

        UV = u_extention*v_extention
        UU = u_extention*u_extention
        VV = v_extention*v_extention
        spectral_uv = fft.fftshift(fft.fft2(UV))[int(NX/2):int(NX*3/2),int(NY/2):int(NY*3/2)]/(NX*NY)
        spectral_uu = fft.fftshift(fft.fft2(UU))[int(NX/2):int(NX*3/2),int(NY/2):int(NY*3/2)]/(NX*NY)
        spectral_vv = fft.fftshift(fft.fft2(VV))[int(NX/2):int(NX*3/2),int(NY/2):int(NY*3/2)]/(NX*NY)

        A2 = u*u+v*v
        LA = 0.01*(1-A2)
        WA = 0.01*(-A2)
        source_x = LA*u-WA*v
        source_y = WA*u+LA*v
        source_x = fft.fftshift(fft.fft2(source_x))/(NX*NY)
        source_y = fft.fftshift(fft.fft2(source_y))/(NX*NY)

        RHS_x = -kx2*spectral_uu-ky2*spectral_uv-nu*(kx1*kx1+ky1*ky1)*spectral_u+source_x
        RHS_y = -kx2*spectral_uv-ky2*spectral_vv-nu*(kx1*kx1+ky1*ky1)*spectral_v+source_y

        spectral_u = spectral_u + RHS_x*deltaT
        spectral_v = spectral_v + RHS_y*deltaT

        u = fft.ifft2(fft.ifftshift(spectral_u))*(NX*NY)
        v = fft.ifft2(fft.ifftshift(spectral_v))*(NX*NY)
        u = np.real(u)
        v = np.real(v)
        sum_u[epoch] = u
        sum_v[epoch] = v

        energy[epoch] = np.sum(u.real**2+v.real**2)

        # print(no, epoch, energy[epoch])
    # }

    print(no)
    # Save dataset
    sum_u[::10].tofile('dataset/dataset_u_'+str(no)+'.dat')
    sum_v[::10].tofile('dataset/dataset_v_'+str(no)+'.dat')

# plt.subplot(221)
# plt.imshow(sum_u[0].real)
# plt.colorbar()
# plt.subplot(222)
# plt.imshow(sum_u[999].real)
# plt.colorbar()
# plt.subplot(223)
# plt.imshow(sum_u[1999].real)
# plt.colorbar()
# plt.subplot(224)
# plt.imshow(sum_u[2999].real)
# plt.colorbar()
# plt.show()

