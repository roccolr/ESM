# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 17:56:22 2025

prova 2, ex 1
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi
plt.close('all')

def block_fun(x, r):
    xc = x[x.shape[0]//2] # valore al centro del blocco
    y = x[np.abs(x-xc)<=r] # elementi nel range
    if len(y)<4:
        v = np.mean(x)
    else:
        v = np.mean(y)
    return v
    
def filtro_sigma(x, K, sigma):
    y = ndi.generic_filter(x, block_fun, (K,K), extra_arguments=(2*sigma,))
    return y

x = np.float64(io.imread('barbara.png'))
M,N = x.shape
sigma = 20
noise = sigma * np.random.randn(M,N)
noisy = x + noise

K = 7
y = filtro_sigma(noisy, K, sigma)

MSE = np.mean((x-y)**2)
PSNR = 10*np.log10(255**2/MSE)
print('PSNR=', PSNR)

plt.figure(1)
plt.subplot(1,3,1)
plt.imshow(x, clim=[0,255], cmap='gray')
plt.title('originale')
plt.subplot(1,3,2)
plt.imshow(noisy, clim=[0,255], cmap='gray')
plt.title('rumorosa')
plt.subplot(1,3,3)
plt.imshow(y, clim=[0,255], cmap='gray')
plt.title('filtrata')

