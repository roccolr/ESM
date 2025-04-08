# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 12:12:01 2025

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

M,N = x.shape
m = np.fft.fftshift(np.fft.fftfreq(M))
n = np.fft.fftshift(np.fft.fftfreq(N))
l,k = np.meshgrid(n,m)
D = np.sqrt(k**2+ l**2)
D0 = 0.1;
H = (D <= D0)
plt.figure(2);
plt.imshow(H, clim=[0,1], cmap='gray',
           extent=(-0.5,+0.5,+0.5,-0.5));


X = np.fft.fft2(x)
X = np.fft.fftshift(X)
plt.figure();
plt.imshow(np.log(1+np.abs(X)), clim=None, cmap='gray', extent=(-0.5,+0.5,+0.5,-0.5));

Y = H * X

plt.figure();
plt.imshow(np.log(1+np.abs(Y)), clim=None, cmap='gray', extent=(-0.5,+0.5,+0.5,-0.5));

Y = np.fft.ifftshift(Y)
y = np.real(np.fft.ifft2(Y))

plt.figure();
plt.imshow(y, clim=[0,255], cmap='gray');
