# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 11:58:19 2025

@author: Davide
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi
plt.close('all')

x = np.float64(io.imread('turbina.jpg'))
plt.figure(1); plt.imshow(x,clim=[0,255],cmap='gray');
plt.title('input')

h = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
y = ndi.correlate(x,h)

plt.figure(2); plt.imshow(y,clim=None,cmap='gray');
plt.colorbar()
plt.title('filtrata')

z = np.abs(y)
th = 0.9*np.max(z)
mask = z<th

plt.figure(3); plt.imshow(mask,clim=None,cmap='gray');
plt.colorbar()
plt.title('maschera')