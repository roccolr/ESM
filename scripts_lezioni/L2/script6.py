# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 10:36:43 2025

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

x_min = np.min(x)
x_max = np.max(x)

#y = x + 50
#y = mylib.fshs(x)
#y = np.log(x+1)
y = x ** 3

plt.figure(2)
plt.imshow(y, clim=None, cmap='gray')
plt.colorbar()

plt.figure(3)
plt.hist(x.flatten(), bins=256)
plt.xlim((0,255))

plt.figure(4)
plt.hist(y.flatten(), bins=256)
plt.xlim((0,255))


