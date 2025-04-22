# -*- coding: utf-8 -*-
"""
@author: %Bombombini Goosini
"""


import numpy as np 
import matplotlib.pyplot as plt
import skimage.io as io 
import scipy.ndimage as ndi 
import skimage.morphology as morph

def fshs(x):
    return (x-np.min(x))/(np.max(x)-np.min(x)) 
    

# def classic_detect(x):
#     h1 = np.array([[0,0,0],[-1,1,0],[0,0,0]])
#     h2 = np.array([[0,-1,0],[0,1,0],[0,0,0]])
    
#     dxv = ndi.correlate(x, h1)
#     dxh = ndi.correlate(x, h2)
#     grad = np.sqrt(dxv**2+dxh**2)
#     mask = grad>0

#     return grad,mask    
    
def new_strategy(x,T):
    k = len(x)
    R = np.mean(x**2)/np.prod(x**2)**(1/(k*k))
    return R>T

def denoise(x):
    x = x/255
    y = x**1.9
    # y = ndi.median_filter(x,3)
    y = ndi.gaussian_filter(x, 2)
    # y = y**0.
    y = ndi.generic_filter(y, np.std, (5,5))
    y = y**0.8
    return y
    
if __name__ == '__main__':
    path = 'C://Users//Flexo Rodriguez//Desktop//ESM/ESM//prove//immagini//'
    im = path + 'target_rumorosa.raw'
    
    x = np.float32(np.fromfile(im, np.float32))
    x = np.reshape(x, (256,256))
    
    y = denoise(x)
    # grad, mask1 = classic_detect(x)
    mask = y>0.043
    
    k = 3
    T= 4000
    z = ndi.generic_filter(x, new_strategy, (k,k), extra_keywords={'T':T})
    
        
    plt.figure(1)
    plt.imshow(x, clim=[0,255], cmap='gray')
    plt.title('input')
    plt.colorbar()
    
    plt.figure(2)
    plt.imshow(y, clim=None, cmap='gray')
    plt.title('denoise')
    plt.colorbar()
    
    plt.figure(3)
    plt.imshow(mask, clim=[0,1], cmap='gray')
    plt.title('maschera classica')
    
    plt.figure(4)
    plt.imshow(z, clim=[0,1], cmap='gray')
    plt.title('maschera nuova strategia')
    
    
    # plt.figure(3)
    # plt.subplot(1,2,2)
    # plt.imshow(grad, clim=None, cmap='gray')
    # plt.title('gradiente')
    # plt.colorbar()
    # plt.subplot(1,2,1)
    # plt.imshow(mask1, clim=[0,1], cmap='gray')
    # plt.title('mask1')