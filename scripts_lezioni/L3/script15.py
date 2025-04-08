# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 11:48:41 2025

@author: Davide
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi
plt.close('all')

x = np.float64(io.imread('luna.jpg'))
plt.figure(1); plt.imshow(x,clim=[0,255],cmap='gray');
plt.title('input')

h1 = np.array([[0,1,0],[1,-4,1],[0,1,0]])
y1 = ndi.convolve(x, h1)
h2 = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
y2 = ndi.convolve(x, h2)

plt.figure(2);
plt.imshow(y1,clim=None,cmap='gray');
plt.colorbar()
plt.title('output 1')
plt.figure(3);
plt.imshow(y2,clim=None,cmap='gray');
plt.colorbar()
plt.title('output 2')

z = x + y2

plt.figure(4);
plt.imshow(z,clim=None,cmap='gray');
plt.colorbar()
plt.title('enh. image')