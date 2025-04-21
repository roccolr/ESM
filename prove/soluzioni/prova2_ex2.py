# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 18:10:14 2025

prova 2, ex 2
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi
plt.close('all')

x = np.float64(io.imread('fiori.jpg'))/255
M,N,K = x.shape

m = np.fft.fftshift(np.fft.fftfreq(M))
n = np.fft.fftshift(np.fft.fftfreq(N))
l,k = np.meshgrid(n,m)
B = 0.15
D = 2*np.abs(np.sqrt(l**2 + k**2) - 0.25)
H = D<B

plt.figure()
plt.imshow(H,clim=[0,1], cmap='gray', extent=(-0.5,+0.5,+0.5,-0.5))

from mpl_toolkits.mplot3d import Axes3D
ax = Axes3D(plt.figure(), auto_add_to_figure=True) # crea una figura per i grafici 3d
ax.plot_surface(l,k,H, linewidth=0, cmap='jet')

R = x[:,:,0]
G = x[:,:,1]
B = x[:,:,2]
fR = np.fft.fftshift(np.fft.fft2(R))
fG = np.fft.fftshift(np.fft.fft2(G))
fB = np.fft.fftshift(np.fft.fft2(B))
fyR = fR * H
fyG = fG * H
fyB = fB * H

yR = np.real(np.fft.ifft2(np.fft.ifftshift(fyR)))
yG = np.real(np.fft.ifft2(np.fft.ifftshift(fyG)))
yB = np.real(np.fft.ifft2(np.fft.ifftshift(fyB)))

y = np.stack((yR,yG,yB),2)
y = (y-np.min(y))/(np.max(y)-np.min(y))

plt.figure()
plt.subplot(1,2,1)
plt.imshow(x)
plt.subplot(1,2,2)
plt.imshow(y)


list_B = [0.05, 0.10, 0.15, 0.20]
list_SNR = list()
for i in range(4):
    B = list_B[i]
    H = D<B
    
    fyR = fR * H
    fyG = fG * H
    fyB = fB * H

    yR = np.real(np.fft.ifft2(np.fft.ifftshift(fyR)))
    yG = np.real(np.fft.ifft2(np.fft.ifftshift(fyG)))
    yB = np.real(np.fft.ifft2(np.fft.ifftshift(fyB)))

    y = np.stack((yR,yG,yB),2)
    MSE = np.mean((x-y)**2)
    SNR = 10*np.log10(np.var(x)/MSE)
    list_SNR.append(SNR)
    
    
plt.figure()
plt.plot(list_B, list_SNR, '-*')
plt.ylabel('SNR')
plt.grid('on')
