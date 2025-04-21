
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 18:10:14 2025

prova 3, ex 2
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi
from skimage.color import rgb2gray
from skimage.feature import canny
plt.close('all')

x = np.float64(io.imread('ala_ape.jpg'))/255

plt.figure()
plt.imshow(x)
plt.title('originale')

y = rgb2gray(x)

mask = canny(y, sigma=2.0, low_threshold=0.1, high_threshold=0.9, use_quantiles=True) 
plt.figure()
plt.imshow(1-mask, clim=[0,1], cmap='gray')
plt.title('bordi')


