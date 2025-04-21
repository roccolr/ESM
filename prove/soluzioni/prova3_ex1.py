# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 18:10:14 2025

prova 3, ex 1
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi
from skimage.color import rgb2gray
plt.close('all')

x1 = rgb2gray(np.float64(io.imread('disk1.jpg'))/255)
x2 = rgb2gray(np.float64(io.imread('disk2.jpg'))/255)

#punto 1
h  = np.array([[0,1,0],[1,-4,1],[0,1,0]])
y1 = ndi.correlate(x1,h)**2
y2 = ndi.correlate(x2,h)**2

#punto 2
y1m = ndi.uniform_filter(y1, (5, 5))
y2m = ndi.uniform_filter(y2, (5, 5))
y1v = ndi.generic_filter(y1, np.var, (5, 5))
y2v = ndi.generic_filter(y2, np.var, (5, 5))
a1  = y1m*y1v
a2  = y2m*y2v

#punto 3 
a1 = a1/(a1+a2+1e-15)
a2 = 1-a1

#punto 4
xf = a1*x1+a2*x2

#show
plt.figure()
plt.subplot(2,3,  1  )
plt.imshow(x1, clim=[0,1], cmap='gray')
plt.title('x_1')
plt.subplot(2,3,  2  )
plt.imshow(x2, clim=[0,1], cmap='gray')
plt.title('x_2')
plt.subplot(2,3,  3  )
plt.imshow(xf, clim=[0,1], cmap='gray')
plt.title('x_f')
plt.subplot(2,3,  4  )
plt.imshow(a1, clim=[0,1], cmap='gray')
plt.title('\alpha_1')
plt.subplot(2,3,  5  )
plt.imshow(a2, clim=[0,1], cmap='gray')
plt.title('\alpha_2')
