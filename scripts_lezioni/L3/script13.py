# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 10:32:31 2025

@author: Davide
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi
plt.close('all')

x = np.float64(io.imread('barbara.png'))
plt.figure(1); plt.imshow(x,clim=[0,255],cmap='gray');
plt.title('input')

M,N = x.shape
d = 25
n = d*np.random.randn(M,N)
noisy = x+n

plt.figure(2); plt.imshow(noisy,clim=[0,255],cmap='gray');
plt.title('noisy')

k = 3
y = ndi.uniform_filter(noisy, (k,k))


plt.figure(3); plt.imshow(y,clim=[0,255],cmap='gray');
plt.title('filtrata')

mse = np.mean((x-y)**2)
print(mse) 