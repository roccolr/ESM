# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %Bombombini Goosini
"""

import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import k_means

def T_opt(x):
    d = np.reshape(x, (-1,1))
    centroids, idx, sum_var = k_means(d, 2)
    return np.mean(centroids)

def thresholding_adattivo(x, L):
    M,N = x.shape
    num_blocchi = M//L
    
    y = np.zeros((M,N), dtype=np.bool_)
    for i in range(num_blocchi):
        blocco = x[i*L:i*L+L, :]
        t = T_opt(blocco)
        mask = blocco > t
        y [i*L:i*L+L, :] = mask
    
    return y

if __name__ == '__main__':
    path = "C://Users//Flexo Rodriguez//Desktop//ESM/ESM//prove//immagini//"
    im1 = path + 'rice.y'
    
    im2 = path + 'rice_bw.y'
    
    x = np.float32(np.reshape(np.fromfile(im1, dtype=np.uint8), (256,256)))
    mask_ideale = np.float32(np.reshape(np.fromfile(im2, dtype=np.uint8), (256,256))) > 0
    
    t = T_opt(x)
    print(t)
    mask_globale = x > t
    
    L_list = []
    
    for i in range(9):
        L_list.append(2**i)
        
    num_corretti = []
    
    for i in range(len(L_list)):
        L = L_list[i]
        y = thresholding_adattivo(x, L)
        num_corretti.append(np.sum(y==mask_ideale))
        
    # perfetto L=16
    
    z = thresholding_adattivo(x, 16)
        
    # sezione stampa 
    
    plt.figure(1)
    plt.imshow(x, clim=[0,255], cmap='gray')
    plt.colorbar()
    plt.title('input')
    
    plt.figure(2)
    plt.imshow(mask_globale, clim=[0,1], cmap='gray')
    plt.colorbar()
    plt.title('segmentazione globale')
    
    plt.figure(3)
    plt.imshow(mask_ideale, clim=[0,1], cmap='gray')
    plt.colorbar()
    plt.title('mask_ideale')
    
    plt.figure(4)
    plt.plot(L_list, num_corretti)
    plt.grid()
    
    plt.figure(5)
    plt.imshow(z, clim=[0,1], cmap='gray')    