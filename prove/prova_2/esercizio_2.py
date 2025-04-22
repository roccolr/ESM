# -*- coding: utf-8 -*-
"""
@author: %Bombombini Goosini
"""


import numpy as np 
import matplotlib.pyplot as plt
import skimage.io as io 
import scipy.ndimage as ndi 

def SNR(x,y):
    return 10*np.log10((np.var(x))/np.mean((x-y)**2))

def band_pass(x,B):
    M,N = x.shape
    X = np.fft.fftshift(np.fft.fft2(x))
    
    # definizione filtro
    mu, nu = (0.25, 0.25)
    m = np.fft.fftshift(np.fft.fftfreq(M))
    n = np.fft.fftshift(np.fft.fftfreq(N))
    l,k = np.meshgrid(n,m)
    
    # D1 = np.sqrt((k+mu)**2 + (l+nu)**2)
    # D2 = np.sqrt((k+mu)**2 + (l-nu)**2)
    # D3 = np.sqrt((k-mu)**2 + (l+nu)**2)
    # D4 = np.sqrt((k-mu)**2 + (l-nu)**2)
    
    # H = (D1<B) | (D2<B) | (D3<B) | (D4<B)
    
    D = 2*(np.abs(np.sqrt(k**2+l**2))-0.25)
    H = D<B
    
    # filtraggio
    Y = X*H 
    y = np.real(np.fft.ifft2(np.fft.ifftshift(Y)))
    return y, X, Y, H

if __name__ == '__main__':
    path = 'C://Users//Flexo Rodriguez//Desktop//ESM/ESM//prove//immagini//'
    im = path + 'fiori.jpg'
    
    x = np.float32(io.imread(im))
    x = x/255
    
    R,G,B = x[:,:,0], x[:,:,1], x[:,:,2]
    
    y_R, X_R, Y_R, H_R = band_pass(R,0.15)
    y_G, X_G, Y_G, H_G = band_pass(G, 0.15)
    y_B, X_B, Y_B, H_B = band_pass(B, 0.15)
    
    y = np.stack((y_R, y_G, y_B), axis=-1)
    
    B_list = [0.05, 0.10, 0.15, 0.20]
    SNR_list = []
    
    for Band in B_list:
        
        y_Rt, X_Rt, Y_Rt, H_Rt = band_pass(R,Band)
        y_Gt, X_Gt, Y_Gt, H_Gt = band_pass(G,Band)
        y_Bt, X_Bt, Y_Bt, H_Bt = band_pass(B,Band)
        y_t = np.stack((y_Rt, y_Gt, y_Bt), axis=-1)
        SNR_list.append(SNR(x,y_t))

        
    
    plt.figure(1)
    plt.imshow(x, clim=[0,1])
    plt.title('input')
    plt.colorbar()
    
    plt.figure(2)
    plt.subplot(1,3,1)
    plt.imshow(R, clim=[0,1], cmap='gray')
    plt.title('R')
    plt.subplot(1,3,2)
    plt.imshow(G, clim=[0,1], cmap='gray')
    plt.title('G')
    plt.subplot(1,3,3)
    plt.imshow(B, clim=[0,1], cmap='gray')
    plt.title('B')
    
    plt.figure(3)
    plt.subplot(1,3,1)
    plt.imshow(np.log(1+np.abs(X_R)), clim=None, cmap='jet', extent = (-0.5, 0.5, 0.5, -0.5))
    plt.title('X_R')
    plt.subplot(1,3,2)
    plt.imshow(np.log(1+np.abs(X_G)), clim=None, cmap='jet', extent = (-0.5, 0.5, 0.5, -0.5))
    plt.title('X_G')
    plt.subplot(1,3,3)
    plt.imshow(np.log(1+np.abs(X_B)), clim=None, cmap='jet', extent = (-0.5, 0.5, 0.5, -0.5))
    plt.title('X_B')  
    
    plt.figure(4)
    plt.subplot(1,3,1)
    plt.imshow(np.log(1+np.abs(Y_R)), clim=None, cmap='jet', extent = (-0.5, 0.5, 0.5, -0.5))
    plt.title('Y_R')
    plt.subplot(1,3,2)
    plt.imshow(np.log(1+np.abs(Y_G)), clim=None, cmap='jet', extent = (-0.5, 0.5, 0.5, -0.5))
    plt.title('Y_G')
    plt.subplot(1,3,3)
    plt.imshow(np.log(1+np.abs(Y_B)), clim=None, cmap='jet', extent = (-0.5, 0.5, 0.5, -0.5))
    plt.title('Y_B')  
    
    plt.figure(5)
    plt.imshow(H_R, clim=None, cmap='gray', extent = (-0.5, 0.5, 0.5, -0.5))
    plt.title('FdT')
    
    plt.figure(6)
    plt.imshow(y, clim=None)
    plt.title('output')
    plt.colorbar()
    
    plt.figure(7)
    plt.plot(B_list, SNR_list)
    plt.grid()
    plt.xlabel("B")
    plt.ylabel("SNR")
    
    
    