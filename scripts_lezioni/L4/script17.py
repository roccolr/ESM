# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 10:55:26 2025

@author: Davide
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi
plt.close('all')

x = np.float64(io.imread('angio.16bit.png'))
plt.figure(1); plt.imshow(x,clim=None,cmap='gray');
plt.title('input')

sigma = 5
y = ndi.gaussian_laplace(x, (sigma,sigma))
from seg_utils import zero_crossing
z = zero_crossing(y)

plt.figure(2); plt.imshow(z,clim=[0,1],cmap='gray');
plt.title('mask')