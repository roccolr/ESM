# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 18:10:14 2025

prova 5, ex 3
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi
import skimage.morphology as morph
plt.close('all')

x = io.imread('auto.jpg')
xf = np.float64(x)/255

plt.figure()
plt.imshow(x)
plt.title('immagine')


list_Q = [10,20,30,40,50,60,70,80,90,100]

plt.figure()
for i in range(len(list_Q)):
    Q = list_Q[i]
    io.imsave('imm.jpg', x, quality=Q)
    xQ = np.float64(io.imread('imm.jpg'))/255
    dQ = (xf - xQ)**2
    dQ = ndi.uniform_filter(np.mean(dQ,2), (16,16))
    plt.subplot(2,5,i+1)
    plt.imshow(dQ, cmap='gray', clim=[0,1e-4])
    plt.title(Q)

# Jpeg ghost maggiormente visibili per Q = 90
Q = 90
io.imsave('imm.jpg', x, quality=Q)
xQ = np.float64(io.imread('imm.jpg'))/255
dQ = (xf - xQ)**2
dQ = ndi.uniform_filter(np.mean(dQ,2), (16,16))

T = 1.5e-5
mask = dQ < T
plt.figure()
plt.imshow(mask,clim=[0,1],cmap='gray')
plt.title('mappa di contraffazione')


# operazioni morfologiche (facoltative)
b = morph.disk(3)
mask = morph.binary_opening(mask, b)
b = morph.disk(20)
mask = morph.binary_closing(mask, b)

plt.figure()
plt.imshow(mask,clim=[0,1],cmap='gray')
plt.title('mappa di contraffazione migliorata')
