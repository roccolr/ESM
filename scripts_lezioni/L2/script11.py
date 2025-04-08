# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 12:15:29 2025

@author: Davide
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 12:10:41 2025

@author: Davide
"""

import numpy as np # importa Numpy
import matplotlib.pyplot as plt # importa Matplotlib
import scipy.ndimage as ndi # importa Scipy per le immagini
import skimage.io as io # importa il modulo Input/Output di SK-Image


plt.close('all')

from skimage.transform import warp

x = np.float32(io.imread('lena.jpg'))
M,N = x.shape
# T = np.array([ [0.5,0,0], [0,0.5,0], [0,0,1]], dtype=np.float32) # ridim.
#T2 = np.array([ [1,0,0], [0,1,0], [100,45.6,1]], dtype=np.float32) # 
T1 = np.array([ [1,0,0], [0,1,0], [M/2,N/2,1]], dtype=np.float32) # 
theta = -np.pi / 3
T2 = np.array([[np.cos(theta),np.sin(theta),0],
              [-np.sin(theta),np.cos(theta),0],
              [0,0,1]], dtype=np.float32)
T3 = np.array([ [1,0,0], [0,1,0], [-M/2,-N/2,1]], dtype=np.float32) # 
T = T3 @ T2 @ T1

A = T[[1,0,2],:][:,[1,0,2]].T

y = warp(x, A, order = 1, cval=128)

plt.subplot(1,2,1);
plt.imshow(x,clim=[0,255],cmap='gray'); plt.title('input');
plt.subplot(1,2,2);
plt.imshow(y,clim=[0,255],cmap='gray'); plt.title('output');
