# -*- coding: utf-8 -*-
"""
@author: %Bombombini Goosini
"""

import numpy as np 
import scipy.ndimage as ndi 
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.color as clr
import skimage.morphology as mor

def laplacian(x):
    h = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=np.float32)
    l = ndi.correlate(x, h)
    return l
    
    
if __name__ == '__main__':
    path = 'C:/Users/Flexo Rodriguez/Desktop/ESM/ESM/prove/immagini/'
    im = path + 'ala_ape.jpg'
    
    x = np.float32(io.imread(im))/255
    x = clr.colorconv.rgb2gray(x) 
    y = -1*x + 1
    y[270:, 600:] = 0
    
    z = ndi.gaussian_filter(y, 5)
    z = z**1.6
    z = laplacian(z)
    mask = z > 0.0007
    mask2 = mor.remove_small_objects(mask, 200)
    mask2 = mor.closing(mask2, mor.disk(1))
    
    plt.figure()
    plt.imshow(y, clim=[0,1], cmap='gray')
    plt.title('originale-contrasto')
    
    plt.figure()
    plt.imshow(z, clim=None, cmap='gray')
    plt.title('elaborazione')
    
    # plt.figure()
    # plt.hist(z.flatten(), bins=255)
    
    plt.figure()
    plt.imshow(mask, clim=[0,1], cmap='gray')
    plt.title('maschera')
    
    plt.figure()
    plt.imshow(mask2, clim=[0,1], cmap='gray')
    plt.title('maschera post opening')
    