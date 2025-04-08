# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 11:31:48 2025

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

k = 7
var_local = ndi.generic_filter(noisy, np.var, (k,k))
mean_local = ndi.generic_filter(noisy, np.mean, (k,k))

y = noisy - (d**2)* (noisy -mean_local ) /var_local


plt.figure(3); plt.imshow(y,clim=[0,255],cmap='gray');
plt.title('output')

mse = np.mean((x-y)**2)
print('MSE=', mse)
