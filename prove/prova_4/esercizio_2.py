# -*- coding: utf-8 -*-
"""
@author: %Bombombini Goosini
"""

import numpy as np 
import matplotlib.pyplot as plt
import scipy.ndimage as ndi 
import skimage.morphology as mf

def custom_f():
    pass

def detect(P1, P2):
    uP1 = ndi.uniform_filter(P1, (127,127))
    uP2 = ndi.uniform_filter(P2, (127,127))
    
    p = ndi.generic_filter((P1-uP1)*(P2-uP2), np.sum, (127,127))/(np.sqrt(ndi.generic_filter((P1-uP1)**2, np.sum, (127,127)))*np.sqrt(ndi.generic_filter((P2-uP2)**2, np.sum, (127,127))))
    mask = p < 0.03
    return p, mask
    
    
if __name__ == '__main__':
    P1 = np.load('C:/Users/Flexo Rodriguez/Desktop/ESM/ESM/prove/immagini/data_P1.npy')
    P2 = np.load('C:/Users/Flexo Rodriguez/Desktop/ESM/ESM/prove/immagini/data_P2.npy')
    im = np.load('C:/Users/Flexo Rodriguez/Desktop/ESM/ESM/prove/immagini/data_img.npy') # immagine a colori
    
    p,mask = detect(P1, P2)
    mask = mf.remove_small_objects(mask, min_size=1024)
    # mask = mf.remove_small_objects(mask, min_size=512)
    
    plt.figure()
    plt.imshow(im, clim=[0,1])
    plt.title('foto ritoccata')
    
    plt.figure()
    plt.imshow(p, clim=None, cmap='jet')
    plt.colorbar()
    plt.title('correlazione')
    
    
    plt.figure()
    plt.imshow(mask, clim=None, cmap='gray')
    plt.colorbar()
    plt.title('maschera')
    