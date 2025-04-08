# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 12:11:13 2025

@author: Davide
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi
plt.close('all')

x = np.float64(io.imread('quadrato.jpg'))
plt.figure(1); plt.imshow(x,clim=[0,255],cmap='gray');
plt.title('input')

h1 = np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]])
h2 = np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]])
h3 = np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]])
h4 = np.array([[-1,-1,2],[-1,2,-1],[2,-1,-1]])

y1 = np.abs(ndi.correlate(x,h1))
y2 = np.abs(ndi.correlate(x,h2))
y3 = np.abs(ndi.correlate(x,h3))
y4 = np.abs(ndi.correlate(x,h4))

plt.figure(2)
plt.imshow(y1, clim=None, cmap='gray')
plt.figure(3)
plt.imshow(y2, clim=None, cmap='gray')
plt.figure(4)
plt.imshow(y3, clim=None, cmap='gray')
plt.figure(5)
plt.imshow(y4, clim=None, cmap='gray')

z = np.stack((y1,y2,y3,y4), -1)
z = np.max(z, -1)

plt.figure(6)
plt.imshow(z, clim=None, cmap='gray')

"""
th1 = 0.9*np.max(y1)
th2 = 0.9*np.max(y2)
th3 = 0.9*np.max(y3)
th4 = 0.9*np.max(y4)

mask1 = y1>th1
mask2 = y2>th2
mask3 = y3>th3
mask4 = y4>th4

plt.figure(3)
plt.subplot(2,2,1)
plt.imshow(mask1, clim=[0,1], cmap='gray')
plt.subplot(2,2,2)
plt.imshow(mask2, clim=[0,1], cmap='gray')
plt.subplot(2,2,3)
plt.imshow(mask3, clim=[0,1], cmap='gray')
plt.subplot(2,2,4)
plt.imshow(mask4, clim=[0,1], cmap='gray')
"""