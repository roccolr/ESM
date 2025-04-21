# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 23:30:15 2025

@author: Flexo Rodriguez
"""

import numpy as np 
import matplotlib.pyplot as plt 
import scipy.ndimage as ndi 
import skimage.io as io 
import skimage.color as clr


if __name__ == '__main__':
    path = "C://Users//Flexo Rodriguez//Desktop//ESM/ESM//prove//immagini//"
    im = path + 'pears_noise.png'
    x = np.float32(io.imread(im))
    x = x/np.max(x)
    y = clr.colorconv.rgb2hsv(x)
    l = y[:,:,2] # luminanza
    L = np.abs(np.fft.fftshift(np.fft.fft2(l)))
    f_L = np.fft.fftshift(np.fft.fft2(l))
    M,N = l.shape
    m = np.fft.fftshift(np.fft.fftfreq(M))
    n = np.fft.fftshift(np.fft.fftfreq(N))
    l,k = np.meshgrid(n,m)
    
    points = [(0.1,0.1), (0.2,0.2), (0.3, 0.3), (0.4, 0.4), (0.5,0.5)]
    radius = 0.015
    
    H = np.ones((M,N), dtype=np.bool_)
    for point in points:
        mu = point[0]
        nu = point[1]
        
        D1 = np.sqrt((k-mu)**2 + (l+nu)**2)
        D2 = np.sqrt((k+mu)**2 + (l-nu)**2) 
        H = H & (D1>radius) & (D2>radius)

    band_h = 0.001
    band_v = 0.001

    R1 = (np.abs(k-points[0][0])>band_h) & (np.abs(l+points[0][1])>band_v)
    R2 = (np.abs(k+points[0][0])>band_h) & (np.abs(l-points[0][1])>band_v)
    R = R1 & R2
    H = H & R
    
    y_l = np.real(np.fft.ifft2(np.fft.ifftshift(H*f_L)))
        
    Hue = y[:,:,0]
    Saturation = y[:,:,1]
    V = y_l
    
    z = np.stack((Hue,Saturation,V), axis=-1)
    z = clr.colorconv.hsv2rgb(z)
    # z = z/np.max(z)
    
    Saturation = Saturation ** 0.4
    
    z2 = clr.colorconv.hsv2rgb(np.stack((Hue,Saturation,V), axis=-1))
    
    plt.figure(1)
    plt.imshow(x, clim=[0,1])
    plt.title('input')
    plt.figure(2)
    plt.imshow(np.log(1+L), clim=None, cmap='jet', extent=(-0.5,0.5,0.5,-0.5))
    plt.title('Trasformata componente luminanza')
    plt.figure(3)
    plt.imshow(H*np.log(1+L), clim=None, cmap='gray', extent=(-0.5,0.5,0.5,-0.5))
    plt.title('FdT')
    plt.figure(4)
    plt.imshow(z, clim=None)
    plt.colorbar()
    plt.title('output 1')
    plt.figure(5)
    plt.imshow(z2, clim=None)
    plt.colorbar()
    plt.title('output 2')