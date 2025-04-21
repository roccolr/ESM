# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 18:10:14 2025

prova 1, ex 1
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi
from skimage.util import random_noise
plt.close('all')

def random_sp(x,p):
    # from skimage.util import random_noise
    # y = random_noise(x/255, mode='s&p', amount=p)*255
    ## in alternativa:
    val = np.random.rand(*x.shape)
    msk1 = val < p
    msk0 = (2*val) < p
    y = np.copy(x)
    y[msk1] = 255
    y[msk0] = 0
    return y

def smf(x,k,T):
    mediana = ndi.median_filter(x,(k, k))
    mask = np.abs(mediana-x)>T
    y = x*(1-mask) + mediana*mask
    return y


x = np.float64(io.imread('lena.jpg'))
noisy = random_sp(x, 0.2)

list_k = [3,5,7,9,11]
list_T = [30,30,30,30,30]

# il valore di soglia T sembra poter essere scelto costante al variare di
# k. Inoltre si nota come, indipendentemente da T

list_smf_PSNR = []
list_med_PSNR = []

for i in range(5):
    k = list_k[i]
    T = list_T[i]
    y1 = smf(noisy,k,T)
    y2 = ndi.median_filter(noisy,(k, k))
    MSE1 = np.mean((y1-x)**2)
    MSE2 = np.mean((y2-x)**2)
    PSNR1 = 10*np.log10(255**2/MSE1)
    PSNR2 = 10*np.log10(255**2/MSE2)
    list_smf_PSNR.append(PSNR1)
    list_med_PSNR.append(PSNR2)
    
    
plt.figure()
plt.plot(list_k, list_med_PSNR, label='med filter')
plt.plot(list_k, list_smf_PSNR, label='smf filter')
plt.ylabel('PSNR')
plt.grid('on')
plt.legend()


k = 5
T = 30
y1 = smf(noisy,k,T)
y2 = ndi.median_filter(noisy,(k, k))

plt.figure()
plt.subplot(1,3,1)
plt.imshow(noisy, clim=[0,255], cmap='gray')
plt.title('rumorosa')
plt.subplot(1,3,2)
plt.imshow(y1, clim=[0,255], cmap='gray')
plt.title('smf')
plt.subplot(1,3,3)
plt.imshow(y2, clim=[0,255], cmap='gray')
plt.title('med')