# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 11:35:09 2025

@author: Davide
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi
from mylib import rgb2cmy
from skimage.color import rgb2hsv, hsv2rgb
plt.close('all')

x = io.imread('colori.jpg')
x = np.float64(x) / 255
plt.figure(); plt.imshow(x);
plt.title('immagine');

#y = rgb2cmy(x)
y = x**4

x_hsv = rgb2hsv(x)
H = x_hsv[:,:,0]
S = x_hsv[:,:,1]
V = x_hsv[:,:,2]

V = V**4

z_hsv = np.stack((H,S,V),-1)
z = hsv2rgb(z_hsv)


plt.figure(); plt.imshow(y);
plt.title('immagine output in RGB');

plt.figure(); plt.imshow(z);
plt.title('immagine output in V');