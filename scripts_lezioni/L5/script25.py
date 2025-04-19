# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 12:16:03 2025

@author: Davide
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi
plt.close('all')

x = np.fromfile('lenarumorosa_verticale.y', np.int16)
x = np.reshape(x, [512,512])
x = np.float64(x)
plt.figure(); plt.imshow(x, clim=[0,256], cmap='gray');
plt.title('immagine rumorosa');
X = np.fft.fftshift(np.fft.fft2(x));
plt.figure();
plt.imshow(np.log(1+np.abs(X)), clim=None, 
           cmap='gray', extent=(-0.5,+0.5,+0.5,-0.5));
plt.title('Trasformata di Fourier immagine rumorosa');

# Definizione del filtro
nu = 0.2;
mu = 0.0;
B = 0.03
B2 = 0.007
B3 = 0.1
m = np.fft.fftshift(np.fft.fftfreq(X.shape[0]))
n = np.fft.fftshift(np.fft.fftfreq(X.shape[1]))
l,k = np.meshgrid(n,m)
D1 = np.sqrt((k-mu)**2+(l-nu)**2)
D2 = np.sqrt((k+mu)**2+(l+nu)**2)
R1 = (np.abs(k-mu) > B2) | (np.abs(l) < B3)
#R2 = np.abs(k+mu) > B2
#R3 = np.abs(l-nu) > B2
#R4 = np.abs(l+nu) > B2
#H = (D1>B) & (D2>B) & R1 & R2 & R3 & R4
H = (D1>B) & (D2>B) & R1
plt.figure();
plt.imshow(H, clim=[0,1], cmap='gray', extent=(-0.5,+0.5,+0.5,-0.5));
plt.title('Riposta in frequenza del filtro');

# Filtraggio
Y = X * H;
plt.figure();
plt.imshow(np.log(1+np.abs(Y)), clim=None, 
           cmap='gray', extent=(-0.5,+0.5,+0.5,-0.5));
plt.title('Trasformata di Fourier immagine filtrata');
y = np.real(np.fft.ifft2(np.fft.ifftshift(Y)));
plt.figure(); plt.imshow(y, clim=[0,256], cmap='gray');
plt.title('Immagine filtrata');

# Calcolo MSE
xo = np.fromfile('lena.y', np.uint8)
xo = np.reshape(xo, [512,512])
xo = np.float64(xo)
#plt.figure(); plt.imshow(xo, clim=[0,256], cmap='gray');
#plt.title('immagine originale');
MSE = np.mean((xo-y) ** 2)
print('MSE=', MSE)
