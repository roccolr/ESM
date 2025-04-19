# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 11:53:46 2025

@author: Davide
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi
from mylib import rgb2cmy, cmy2rgb
from skimage.color import rgb2hsv, hsv2rgb
plt.close('all')

x = io.imread('foto.jpg')
x = np.float64(x) / 255
plt.figure(); plt.imshow(x);
plt.title('immagine');

x_cmy = rgb2cmy(x)
C = x_cmy[:,:,0]
M = x_cmy[:,:,1]
Y = x_cmy[:,:,2]

C = C ** 1.5

y_cmy = np.stack((C,M,Y),-1)
y = cmy2rgb(y_cmy)

plt.figure(); plt.imshow(y);
plt.title('output');
