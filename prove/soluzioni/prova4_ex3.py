# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 17:56:22 2025

prova 4, ex 3
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi
plt.close('all')

x1 = np.float64(io.imread('img1.png'))
x2 = np.float64(io.imread('img2.png'))


m1 = ndi.gaussian_filter(x1, (10,10))<15
m2 = ndi.gaussian_filter(x2, (10,10))<15

plt.figure()
plt.subplot(1,2,1)
plt.imshow(x1, clim=[0,255], cmap='gray')
plt.title('immagine')
plt.subplot(1,2,2)
plt.imshow(m1, clim=[0,1], cmap='gray')
plt.title('risultato')

plt.figure()
plt.subplot(1,2,1)
plt.imshow(x2, clim=[0,255], cmap='gray')
plt.title('immagine')
plt.subplot(1,2,2)
plt.imshow(m2, clim=[0,1], cmap='gray')
plt.title('risultato')
