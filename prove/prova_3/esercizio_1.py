# -*- coding: utf-8 -*-
"""
@author: %Bombombini Goosini
"""

import numpy as np
import scipy.ndimage as ndi
import skimage.io as io
import matplotlib.pyplot as plt
import skimage.color as clr

def laplaciano(x):
    h1 = np.array([[0,-1,0],[0,2,0],[0,-1,0]], dtype=np.float32)
    h2 = np.array([[0,0,0],[-1,2,-1], [0,0,0]], dtype=np.float32)
    # h = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)
    
    d_x_m = ndi.convolve(x, h1)
    d_x_n = ndi.convolve(x, h2)
    
    laplaciano = d_x_m + d_x_n
    
    return laplaciano

def activity(x):
    loc_mean = ndi.uniform_filter(x, 5)
    loc_var = ndi.generic_filter(x, np.var, (5,5))
    
    return loc_mean * loc_var

def normalize(a1,a2):
    return a1/(a1+a2+1e-15), 1-a1/(a1+a2+1e-15)

def fusion(x1, x2, a1, a2):
    return a1*x1 + a2*x2

if __name__ == '__main__':
    path = 'C:/Users/Flexo Rodriguez/Desktop/ESM/ESM/prove/immagini/'
    im1 = path + 'disk1.jpg'
    im2 = path + 'disk2.jpg'
    
    x1 = np.float32(io.imread(im1))/255
    x2 = np.float32(io.imread(im2))/255

    x1 = clr.colorconv.rgb2gray(x1)
    x2 = clr.colorconv.rgb2gray(x2)
    
    
    l1, l2 = ((laplaciano(x1)**2), (laplaciano(x2)**2))
    a1, a2 = activity(l1), activity(l2)
    a1, a2 = normalize(a1,a2)
    
    f = fusion(x1, x2, a1, a2)
    
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.imshow(x1, clim=None, cmap='gray')
    plt.title('immagine 1')
    plt.subplot(1,2,2)
    plt.imshow(x2, clim=None, cmap='gray')
    plt.title('immagine 2')
    plt.figure(2)
    plt.imshow(f, clim=None, cmap='gray')
    plt.title('fusione')

        