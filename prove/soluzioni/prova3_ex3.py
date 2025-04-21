# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 18:10:14 2025

prova 3, ex 3
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi
from skimage.color import rgb2gray
plt.close('all')

x1 = rgb2gray(np.float64(io.imread('I1.png'))/255)
x2 = rgb2gray(np.float64(io.imread('I2.png'))/255)

plt.figure()
plt.imshow(x1, clim=[0,1], cmap='gray')
plt.title('I1')

plt.figure()
plt.imshow(x2, clim=[0,1], cmap='gray')
plt.title('I2')

def block_lpb(x):
    x = np.reshape(x,(3,3))
    x = (x - x[1,1])>=0
    y = x[0,0]+2*x[0,1]+4*x[0,2]+8*x[1,2]+16*x[2,2]+32*x[2,1]+64*x[2,0]+128*x[1,0]
    return y


h = np.array([[-1,2,-1],[2,-4,2],[-1,2,-1]])
y1 = ndi.correlate(x1,h)
y2 = ndi.correlate(x2,h)
    
z1 = ndi.generic_filter(y1, block_lpb, (3,3))
z2 = ndi.generic_filter(y2, block_lpb, (3,3))
    
h1, b1 = np.histogram(z1,np.arange(257))
h2, b2 = np.histogram(z2,np.arange(257))
s1 = np.std(h1)
s2 = np.std(h2)

if s1>495:
    print('I1 è vera')
else:
    print('I1 è false')
    
if s2>495:
    print('I2 è vera')
else:
    print('I2 è false')
