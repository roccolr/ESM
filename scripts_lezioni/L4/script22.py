# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 12:07:42 2025

@author: Davide
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi
plt.close('all')


x = np.float64(io.imread('lena.jpg'))
plt.figure(1); plt.imshow(x,clim=[0,255],cmap='gray');
plt.title('input')


h = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=np.float64)
M,N = x.shape
A,B = h.shape
P = M 
Q = N 
X = np.fft.fft2(x, (P,Q))
H = np.fft.fft2(h, (P,Q))
Y = H * X
y = np.real(np.fft.ifft2(Y))
plt.figure(); plt.imshow(y, clim=None, cmap='gray');
