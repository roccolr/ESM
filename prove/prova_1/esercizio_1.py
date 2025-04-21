# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 22:19:16 2025

@author: Flexo Rodriguez
"""

import numpy as np 
import matplotlib.pyplot as plt 
import skimage.io as io 
from skimage.util import random_noise
import scipy.ndimage as ndi


def add_noise(x, p):
    x = x/np.max(x)
    y = random_noise(x, mode='s&p', salt_vs_pepper=p)
    return y

def smf(x,k,T):
    m = ndi.generic_filter(x, np.median, (k,k))
    z = np.abs(m-x) > T
    y = z*m + x*(1-z)
    return y

def MSE(x,y):
    return np.mean((x-y)**2)

def PSNR(x,y):
    max_x = np.max(x)
    return 10*np.log10((max_x**2)/MSE(x,y))
    

if __name__ == '__main__':
    plt.close('all')
    path = "C://Users//Flexo Rodriguez//Desktop//ESM/ESM//prove//immagini//"
    x = np.float32(io.imread(path+'lena.jpg'))
    noisy_lena = add_noise(x, 0.2)
    
    values_k = [3,5,7,9,11]
    y = smf(x , 5, 30)
    
    PSNR_1 = []
    PSNR_2 = []
    
    for value in values_k:
        PSNR_1.append(PSNR(x, smf(noisy_lena, value, 2)))
        PSNR_2.append(PSNR(x, ndi.median_filter(noisy_lena, value, mode='reflect')))
        
    
    
    plt.figure(1)
    plt.imshow(x, clim=None, cmap='gray')
    plt.title('input')
    plt.colorbar()
    plt.figure(2)
    plt.imshow(noisy_lena, clim=None, cmap='gray')
    plt.title('noisy_lena')
    plt.colorbar()
    plt.figure(3)
    plt.imshow(y, clim=[0,255], cmap='gray')
    plt.title('output')
    plt.colorbar()
    # plt.figure(4)
    # plt.plot(values_k,PSNR_1)
    # plt.title('PSNR metodo adattivo')
    # plt.figure(5)
    # plt.plot(values_k,PSNR_2)
    # plt.title('PSNR metodo standard')
    plt.show()