# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 11:58:29 2025

@author: Davide
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi
from mylib import rgb2cmy, cmy2rgb
from skimage.color import rgb2hsv, hsv2rgb
from color_convertion import rgb2hsi, hsi2rgb
plt.close('all')

x = io.imread('Azzurro.jpg')
x = np.float64(x) / 255
plt.figure(); plt.imshow(x);
plt.title('immagine');

x_hsv = rgb2hsi(x)
H = x_hsv[:,:,0]
S = x_hsv[:,:,1]
V = x_hsv[:,:,2]

plt.figure()
plt.subplot(1,3,1)
plt.imshow(H, clim=[0,1], cmap='gray')
plt.title('hue')
plt.subplot(1,3,2)
plt.imshow(S, clim=[0,1], cmap='gray')
plt.title('sat.')
plt.subplot(1,3,3)
plt.imshow(V, clim=[0,1], cmap='gray')
plt.title('value')

mask = (H<0.62) & (H>0.57) & (S>0.4)
yH = 0 #(H-0.6)%1
yH = mask*yH+(1-mask)*H
y_hsv = np.stack((yH,S,V),-1)
y = hsi2rgb(y_hsv)

 
plt.figure()
plt.imshow(yH, clim=[0,1], cmap='gray')
plt.title('yH')

plt.figure()
plt.imshow(mask, clim=[0,1], cmap='gray')
plt.title('mask')

plt.figure()
plt.imshow(y)
plt.title('output')
