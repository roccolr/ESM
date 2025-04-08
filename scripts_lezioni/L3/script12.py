# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 10:32:31 2025

@author: Davide
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi

x = np.float64(io.imread('space.jpg'))
plt.figure(1); plt.imshow(x,clim=[0,255],cmap='gray');
plt.title('input')

#k = 11; h = np.ones((k,k))/(k**2)
#y = ndi.correlate(x, h, mode='reflect')
#y = ndi.gaussian_filter(x, 3)
#h = np.array([[1,2,1],[2,4,2],[1,2,1]])/16
#y = ndi.correlate(x, h, mode='reflect')

x_filt = ndi.uniform_filter(x, (15,15))
th = 0.25*np.max(x_filt)
#mask = x_filt>th
#y = mask*x
x[x_filt<th] = 0

plt.figure(2); plt.imshow(x,clim=[0,255],cmap='gray');
plt.title('output')