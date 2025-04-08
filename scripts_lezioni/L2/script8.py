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

plt.close('all')

x = io.imread('lena.jpg')

plt.figure(1)
plt.imshow(x, clim=[0,255], cmap='gray')
plt.title('input')

plt.figure(2)
for index in range(8):
    B = bitget(x, index) # estrazione bit-plane pi`u significativo
    plt.subplot(2,4,index+1)
    plt.imshow(B, clim=[0,1], cmap='gray') # visualizzazione del bit-plane
    plt.title('bit-plane %d'%index)
    

