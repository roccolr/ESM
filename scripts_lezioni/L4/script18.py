# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 10:58:38 2025

@author: Davide
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi
plt.close('all')

x = np.float64(io.imread('circuito_rumoroso.jpg'))
y = ndi.generic_filter(x, np.median, (5,5))
plt.subplot(1,2,1); plt.imshow(x,clim=[0,255],cmap='gray');
plt.subplot(1,2,2); plt.imshow(y,clim=[0,255],cmap='gray');
