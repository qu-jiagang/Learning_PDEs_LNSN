import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft


# Constant parameters

nu = 0.015
NX,NY = 128,128
LX,LY = 2*np.pi,2*np.pi
deltaX = LX/NX
deltaY = LY/NY
deltaT = 0.001
iter_step = 3001

print(1-2*deltaT/deltaX/deltaX*nu-2*deltaT/deltaY/deltaY*nu)
# exit()

for no in range(0,1):
    # Initial condition
    # {
    kx1 = np.linspace(-NX/2,NX/2,NX,endpoint=False).reshape(NX,1)
    kx2 = np.linspace(-NX/2,NX/2,NX,endpoint=False).reshape(NX,1)*1j
    ky1 = kx1.T
    ky2 = kx2.T

    u,v = np.zeros([NX,NY]),np.zeros([NX,NY])
    for i in range(NX):
        for j in range(NY):
            for w in range(9):
                for l in range(9):
                    u[i,j] += np.random.randn()*np.cos((w-4)*i*deltaX+(l-4)*i*deltaX) +\
                              np.random.randn()*np.sin((w-4)*i*deltaX+(l-4)*i*deltaX)
                    v[i,j] += np.random.randn()*np.cos((w-4)*i*deltaX+(l-4)*i*deltaX) +\
                              np.random.randn()*np.sin((w-4)*i*deltaX+(l-4)*i*deltaX)

    u = 2*np.real(u)/np.max(np.real(u))+(np.random.random([NX,NY])-0.5)*4
    v = 2*np.real(v)/np.max(np.real(v))+(np.random.random([NX,NY])-0.5)*4

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

        spectral_u = fft.fftshift(fft.fft2(u)) / (NX * NY)
        spectral_v = fft.fftshift(fft.fft2(v)) / (NX * NY)

    u = np.sqrt(1/np.mean(u*u))*u*1
    v = np.sqrt(1/np.mean(v*v))*v*1
    spectral_u = fft.fftshift(fft.fft2(u)) / (NX * NY)
    spectral_v = fft.fftshift(fft.fft2(v)) / (NX * NY)
    # }

    # Allocate some variables
    # {
    energy = np.zeros(iter_step)
    sum_u = np.zeros([iter_step,NX,NY])
    sum_v = np.zeros([iter_step,NX,NY])
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

        RHS_x = -kx2*spectral_uu-ky2*spectral_uv-nu*(kx1*kx1+ky1*ky1)*spectral_u
        RHS_y = -kx2*spectral_uv-ky2*spectral_vv-nu*(kx1*kx1+ky1*ky1)*spectral_v

        spectral_u = spectral_u + RHS_x*deltaT
        spectral_v = spectral_v + RHS_y*deltaT

        u = fft.ifft2(fft.ifftshift(spectral_u))*(NX*NY)
        v = fft.ifft2(fft.ifftshift(spectral_v))*(NX*NY)

        sum_u[epoch] = np.real(u)
        sum_v[epoch] = np.real(v)

        energy[epoch] = np.sum(u.real**2+v.real**2)

        # print(epoch,energy[epoch])
    # }

    # Save dataset
    print(no)
    sum_u[::10].tofile('dataset/dataset_u_'+str(no)+'.dat')
    sum_v[::10].tofile('dataset/dataset_v_'+str(no)+'.dat')

plt.subplot(221)
plt.imshow(sum_u[0].real)
plt.colorbar()
plt.subplot(222)
plt.imshow(sum_u[99].real)
plt.colorbar()
plt.subplot(223)
plt.imshow(sum_u[199].real)
plt.colorbar()
plt.subplot(224)
plt.imshow(sum_u[299].real)
plt.colorbar()
plt.show()

