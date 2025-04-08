# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 11:16:56 2025

@author: Davide
"""

import numpy as np # importa Numpy
import matplotlib.pyplot as plt # importa Matplotlib
import scipy.ndimage as ndi # importa Scipy per le immagini
import skimage.io as io # importa il modulo Input/Output di SK-Image
import mylib
plt.close('all')

x = io.imread('filamento.jpg')
x = np.float64(x)

plt.figure(1)
plt.imshow(x, clim=[0,255], cmap='gray')
plt.title('input')

med = np.mean(x) # media globale
dev = np.std(x)  # std globale
MED = ndi.generic_filter(x, np.mean, (3,3)) # media locale 
DEV = ndi.generic_filter(x, np.std, (3,3))  # std locale

plt.figure(2)
plt.imshow(MED, clim=None, cmap='gray')
plt.title('medie locali')

plt.figure(3)
plt.imshow(DEV, clim=None, cmap='gray')
plt.title('std locali')

mask = (MED<=0.4*med) & (DEV<=0.4*dev) & (DEV>=0.02*dev)
plt.figure(4)
plt.imshow(mask, clim=[0,1], cmap='gray')
plt.title('mask')


#y = mask*(4*x) + (1-mask)*x
y = np.copy(x)
y[mask] = 4*x[mask]


plt.figure(5)
plt.imshow(y, clim=[0,255], cmap='gray')
plt.title('output')
