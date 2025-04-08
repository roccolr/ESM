# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 11:50:35 2025

@author: Davide
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 11:39:47 2025

@author: Davide
"""

import numpy as np # importa Numpy
import matplotlib.pyplot as plt # importa Matplotlib
import scipy.ndimage as ndi # importa Scipy per le immagini
import skimage.io as io # importa il modulo Input/Output di SK-Image
import mylib
from bitop import bitget
from bitop import bitset

plt.close('all')

x = np.fromfile('lena.y', np.uint8)
x = np.reshape(x, (512,512))
x = x[0:350,0:350]

plt.figure(1)
plt.imshow(x, clim=[0,255], cmap='gray')
plt.title('input')

m = np.fromfile('marchio.y', np.uint8)
m = np.reshape(m, (350,350))

plt.figure(2)
plt.imshow(m, clim=None, cmap='gray')
plt.title('marchio')

y = bitset(x,0,m)

plt.figure(3)
plt.imshow(y, clim=[0,255], cmap='gray')
plt.title('output')

plt.figure(4)
plt.hist(x.flatten(), bins=256)
plt.xlim((0,255))

plt.figure(5)
plt.hist(y.flatten(), bins=256)
plt.xlim((0,255))
