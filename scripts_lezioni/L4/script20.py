# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 11:35:28 2025

@author: Davide
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi
plt.close('all')


#x = np.float64(io.imread('rettangolo.jpg'))
#plt.figure(1); plt.imshow(x,clim=[0,255],cmap='gray');
#plt.title('input')
k=3; h = np.ones((k,k))/(k**2)

#X = np.fft.fft2(x)
P = 100; Q = 100
X = np.fft.fft2(h, (P,Q))

plt.figure(2); plt.imshow(np.abs(X), clim=None, cmap='gray');

Y = np.log(1+np.abs(np.fft.fftshift(X)))
plt.figure(3); plt.imshow(Y, clim=None, cmap='gray',
                          extent=(-0.5,+0.5,+0.5,-0.5));

m = np.fft.fftshift(np.fft.fftfreq(Y.shape[0]))
n = np.fft.fftshift(np.fft.fftfreq(Y.shape[1]))
from mpl_toolkits.mplot3d import Axes3D
ax = Axes3D(plt.figure()); # crea una figura per i grafici 3d
l,k = np.meshgrid(n,m)
ax.plot_surface(l,k,Y, linewidth=0, cmap='jet')