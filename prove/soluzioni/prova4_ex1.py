# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 17:56:22 2025

prova 4, ex 1
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi
from skimage.feature import canny
import skimage.morphology as morph
from bitop import bitset,bitget
plt.close('all')


x = np.reshape(np.fromfile('upupa.y', np.uint8), (256,512))
firma = np.reshape(np.fromfile('firma.y', np.uint8), (256,512))
firma = firma>0

plt.figure()
plt.imshow(x,clim=[0,255], cmap='gray')
plt.title('immagine')

plt.figure()
plt.imshow(firma,clim=[0,1], cmap='gray')
plt.title('firma')


# inserisce firma
xw = bitset(x, 1, firma)

# calcolo mse firma compressa
list_Q = [80, 90, 100]
list_MSE = []
for i in range(3):
    Q = list_Q[i]
    io.imsave('xw.jpeg', xw, quality=Q)
    xwjpg = io.imread('xw.jpeg')
    firma1 = bitget(xwjpg, 1)
    MSE = np.mean((np.float32(firma) - np.float32(firma1))**2)
    list_MSE.append(MSE)

# visualizzazione mse
plt.figure()
plt.plot(list_Q, list_MSE, 'o-')
plt.xlabel('Q')
plt.ylabel('MSE')
plt.grid('on')
plt.title('compressione')


# calcolo mse filtro lp
list_D0 = [0.2, 0.3, 0.4]
list_MSE = []
X = np.fft.fft2(xw)
m = np.fft.fftfreq(X.shape[0])
n = np.fft.fftfreq(X.shape[1])
l,k = np.meshgrid(n,m)
dist = np.sqrt(l**2 + k**2)
for i in range(3):
    D0 = list_D0[i]
    H = dist<D0
    Y = X*H
    y = np.uint8(np.real(np.fft.ifft2(Y)))
    firma1 = bitget(y, 1)
    MSE = np.mean((np.float32(firma) - np.float32(firma1))**2)
    list_MSE.append(MSE)

# visualizzazione mse
plt.figure()
plt.plot(list_D0, list_MSE, 'o-')
plt.xlabel('D0')
plt.ylabel('MSE')
plt.grid('on')
plt.title('filtraggio')
