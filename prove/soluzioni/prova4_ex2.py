# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 17:56:22 2025

prova 4, ex 2
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi
plt.close('all')

def bloc_fun_den(x):
    y = (x-np.mean(x))**2
    y = np.sqrt(np.sum(y))
    return y

def detect(P1,P2):
    medie1 = ndi.uniform_filter(P1, (127, 127))
    medie2 = ndi.uniform_filter(P2, (127, 127))
    corr12 = ndi.uniform_filter(P1*P2, (127, 127))
    num = (corr12-medie1*medie2)*127*127
    den1 = ndi.generic_filter(P1, bloc_fun_den, (127, 127))
    den2 = ndi.generic_filter(P2, bloc_fun_den, (127, 127))
    mappa = num/(den1*den2)
    mask = mappa<0.02
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(mappa, cmap='jet', clim=None)
    plt.colorbar()
    plt.title('Correlazione')
    plt.subplot(1,2,2)
    plt.imshow(mask, clim=(0,1), cmap='gray')
    plt.title('Maschere')
    return mask

P1 = np.load('data_P1.npy')
P2 = np.load('data_P2.npy')
img = np.load('data_img.npy')

plt.figure()
plt.imshow(img)
plt.title('immagine')

mask = detect(P1,P2)

