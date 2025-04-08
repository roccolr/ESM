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

x = np.fromfile('lenarumorosa.y', np.int16)
x = np.reshape(x, [512,512])
x = np.float64(x)
plt.figure(); plt.imshow(x, clim=[0,256], cmap='gray');
plt.title('immagine rumorosa');
X = np.fft.fftshift(np.fft.fft2(x));
plt.figure();
plt.imshow(np.log(1+np.abs(X)), clim=None, 
           cmap='gray', extent=(-0.5,+0.5,+0.5,-0.5));
plt.title('Trasformata di Fourier immagine rumorosa');