# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 11:47:45 2025

@author: Davide
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi
plt.close('all')


x = np.float64(io.imread('volto.tif'))
plt.figure(1); plt.imshow(x,clim=[0,255],cmap='gray');
plt.title('input')

X = np.fft.fftshift(np.fft.fft2(x))

plt.figure(2); plt.imshow(np.log(1+np.abs(X)),
                          clim=None, cmap='gray',
                          extent=(-0.5,+0.5,+0.5,-0.5))

plt.figure(3); plt.imshow(np.angle(X),
                          clim=[-np.pi,np.pi], cmap='gray',
                          extent=(-0.5,+0.5,+0.5,-0.5))

X1 = np.abs(X)
X2 = np.exp(1j*np.angle(X))

y1 = np.real(np.fft.ifft2(np.fft.ifftshift(X1)))
y2 = np.real(np.fft.ifft2(np.fft.ifftshift(X2)))

plt.figure(4); plt.imshow((y1-np.min(y1))**0.1, clim=None, cmap='gray')
plt.figure(5); plt.imshow(y2, clim=None, cmap='gray')
