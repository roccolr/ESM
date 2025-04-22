# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %Bombombini Goosini
"""

import numpy as np 
import matplotlib.pyplot as plt
import skimage.io as io 
import scipy.ndimage as ndi 


def filtro_sigma(x, k, sigma):
    # viene passata la finestra kxk
    
    if k<4:
        return np.mean(x)
    else:
        current_pixel = x[len(x)//2 + 1]
        mask = np.abs(x-current_pixel)<(2*sigma)
        return np.mean(x[mask])        
        
    
    
def add_noise(x, sigma):
    noise = sigma*np.random.randn(x.shape[0], x.shape[1])
    return x+noise

def MSE(x,y):
    return np.mean((x-y)**2)

def PSNR(x,y):
    max_x = np.max(x)
    return 10*np.log10((max_x**2)/MSE(x,y))

if __name__ == '__main__':
    path = 'C://Users//Flexo Rodriguez//Desktop//ESM/ESM//prove//immagini//'
    im = path + 'barbara.png'
    
    x = np.float32(io.imread(im))
    k = 7
    sigma = 20
    noisy_x = add_noise(x, sigma)
    y = ndi.generic_filter(noisy_x, filtro_sigma, (k,k),extra_keywords={'k':k, 'sigma':sigma})
    
    psnr = PSNR(x, y)
    print(f'[PSNR] = {psnr}')
    
    plt.figure(1)
    plt.imshow(x, clim=[0,255], cmap='gray')
    plt.title('input')
    plt.colorbar()
    plt.figure(2)
    plt.imshow(noisy_x, clim=[0,255], cmap='gray')
    plt.title('noisy_input')
    plt.colorbar()
    plt.figure(3)
    plt.imshow(y, clim=[0,255], cmap='gray')
    plt.title('output')
    plt.colorbar()