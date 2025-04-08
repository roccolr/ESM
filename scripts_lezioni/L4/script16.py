# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 10:35:39 2025

@author: Davide
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi
plt.close('all')

x = np.float64(io.imread('angiogramma.jpg'))
plt.figure(1); plt.imshow(x,clim=[0,255],cmap='gray');
plt.title('input')

#h1 = np.array([[0,0],[-1,1]])
#h2 = np.array([[0,-1],[0,1]])
h1 = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
h2 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

y = ndi.gaussian_filter(x, (3,3))
y1 = ndi.correlate(y, h1)
y2 = ndi.correlate(y, h2)

g = np.sqrt(y1**2+y2**2)

plt.figure(2); plt.imshow(g,clim=None,cmap='gray');
plt.title('gradiente')

T = 1.5*np.mean(g)

msk = g>T

plt.figure(3); plt.imshow(msk,clim=[0,1],cmap='gray');
plt.title('maschera di seg.')
 