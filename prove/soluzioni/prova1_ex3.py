# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 18:10:14 2025

prova 1, ex 3
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi
from sklearn.cluster import k_means
plt.close('all')

def T_opt(x):
    d = np.reshape(x, (-1,1))
    centroids, idx, var_sum = k_means(d, 2)
    T = np.mean(centroids)
    return T

def mask_adap(x, L):
    M,N = x.shape
    num_block = M//L
    y = np.zeros((M,N), np.bool_)
    for j in range(num_block):
        block = x[j*L:(j*L+L),:]
        mask_block = block > T_opt(block)
        y[j*L:(j*L+L),:] = mask_block
    return y
    

x = np.reshape(np.fromfile('rice.y', np.uint8), (256,256))
x = np.float64(x)

mask_ideal = np.reshape(np.fromfile('rice_bw.y', np.uint8), (256,256))
mask_ideal = mask_ideal>0

plt.figure()
plt.imshow(x,clim=[0,255], cmap='gray')
plt.title('immagine originale')

plt.figure()
plt.imshow(mask_ideal,clim=[0,1], cmap='gray')
plt.title('maschera ideale')

t = T_opt(x)
mask = x > t

plt.figure()
plt.imshow(mask,clim=[0,1], cmap='gray')
plt.title('immagine con segmentazione globale')


list_L = [1,2,4,8,16,32,64,128,256]
list_correct = []
for i in range(len(list_L)):
    L = list_L[i]
    y = mask_adap(x, L)
    num_correct = np.sum(y==mask_ideal)
    list_correct.append(num_correct)

plt.figure()
plt.semilogx(list_L, list_correct, '-*')
plt.grid('on')
plt.ylabel('pixel corretti')

# ottimo con L = 16
L = 16
y = mask_adap(x, L)

plt.figure()
plt.imshow(y,clim=[0,1], cmap='gray')
plt.title(f'immagine con segmentazione locale L={L}')
