# -*- coding: utf-8 -*-
"""
@author: %Bombombini Goosini
"""

import numpy as np 
import matplotlib.pyplot as plt
import scipy.ndimage as ndi 
import skimage.io as io 
    
    
if __name__ == '__main__':
    path = 'C:/Users/Flexo Rodriguez/Desktop/ESM/ESM/prove/immagini/'
    x1 = np.float32(io.imread(path+'img1.png'))
    x2 = np.float32(io.imread(path+'img2.png'))
    
    y1 = ndi.gaussian_filter(x1, 10)
    y2 = ndi.gaussian_filter(x2, 10)    
    
    avg_1 = np.mean(y1)
    avg_2 = np.mean(y2)
    
    map_1 = y1 < 0.25*avg_1
    map_2 = y2 < 0.25*avg_2
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(x1, clim=[0,255], cmap='gray')
    plt.title('img1.png')
    plt.subplot(1,2,2)
    plt.imshow(x2, clim=[0,255], cmap='gray')
    plt.title('img2.png')

    
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(y1, clim=[0,255], cmap='gray')
    # plt.title('elab_1')
    # plt.subplot(1,2,2)
    # plt.imshow(y2, clim=[0,255], cmap='gray')
    # plt.title('elab_2')
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(map_1, clim=[0,1], cmap='gray')
    plt.title('map_1')
    plt.subplot(1,2,2)
    plt.imshow(map_2, clim=[0,1], cmap='gray')
    plt.title('map_2')