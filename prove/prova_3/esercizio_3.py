# -*- coding: utf-8 -*-
"""
@author: %Bombombini Goosini
"""

import numpy as np
import scipy.ndimage as ndi
import skimage.io as io
import matplotlib.pyplot as plt
import skimage.color as clr

def calcolo_x(z):
    h = np.array([[-1,2,-1],[2,-4,2],[-1,2,-1]])
    x = ndi.correlate(z, h)
    return x

def u(x):
    if x>=0:
        return 1
    else:
        return 0
    
def custom_f(x):
    x = np.reshape(x, (3, 3))
    x_c = x[1,1]
    y = u(x[0,0]-x_c)
    y += u(x[0,1]-x_c)*2
    y += u(x[0,2]-x_c)*4
    y += u(x[1,2]-x_c)*8
    y += u(x[2,2]-x_c)*16
    y += u(x[2,1]-x_c)*32
    y += u(x[2,0]-x_c)*64
    y += u(x[1,0]-x_c)*128
    return y

    

if __name__ == '__main__':
    path = 'C:/Users/Flexo Rodriguez/Desktop/ESM/ESM/prove/immagini/'
    im1 = path + 'I1.png'
    im2 = path + 'I2.png'
    
    z1 = np.float64(io.imread(im1))/255
    z2 = np.float64(io.imread(im2))/255
    z1 = clr.colorconv.rgb2gray(z1)
    z2 = clr.colorconv.rgb2gray(z2)

    
    x1, x2 = calcolo_x(z1), calcolo_x(z2)
    y1 = ndi.generic_filter(x1, custom_f, (3,3))
    y2 = ndi.generic_filter(x2, custom_f, (3,3))
    
    esito1 = 'false'
    esito2 = 'false'
    h1, b1 = np.histogram(y1,np.arange(257))
    h2, b2 = np.histogram(y2,np.arange(257))
    s1 = np.std(h1)
    s2 = np.std(h2)
    if s1>495:
        esito1 = 'true'
    if s2>495:
        esito2 = 'true'
        
    print(s1,s2)
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(z1, clim=None, cmap='gray')
    plt.title(f'input 1 - {esito1}')
    plt.subplot(1,2,2)
    plt.imshow(z2, clim=None, cmap='gray')
    plt.title(f'input 2 - {esito2}')
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.hist(y1.flatten(),bins=255)
    plt.title('hist 1')
    plt.subplot(1,2,2)
    plt.hist(y2.flatten(),bins=255)
    plt.title('hist 2')
    
    
    
    
    