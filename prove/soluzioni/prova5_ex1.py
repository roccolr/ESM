# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 18:10:14 2025

prova 5, ex 1
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi
import skimage.morphology as morph
plt.close('all')

h = np.array([[1,0,0,0,1], [0,0,0,0,0], [0,0,-4,0,0],
              [0,0,0,0,0], [1,0,0,0,1]])

# punto 1 : risposta in frequenza
def respfreq(h, M=100, N=100):
    m = np.fft.fftshift(np.fft.fftfreq(M))
    n = np.fft.fftshift(np.fft.fftfreq(N))
    l,k = np.meshgrid(n,m)
    hh = np.rot90(h,2) # ribaltamento della maschera
    H = np.fft.fftshift(np.fft.fft2(hh,s=(M,N)))
    

    from mpl_toolkits.mplot3d import Axes3D
    ax = Axes3D(plt.figure(), auto_add_to_figure=True) # crea una figura per i grafici 3d
    ax.plot_surface(l,k,np.abs(H), linewidth=0, cmap='jet')
    plt.title("risposta in frequenza")

    return H

H = respfreq(h)

# punto 2: filtraggio spaziale
def filtra(x,h):
    y = ndi.correlate(x, h, mode='constant')
    return y

x = np.float64(io.imread('barbara.png'))

y1 = filtra(x,h)




#punto 3: filtraggio in frequenza
def filtrafreq(x,h):
    M,N = x.shape
    P,Q = h.shape
    Mf = M+P-1
    Nf = N+Q-1
    X = np.fft.fft2(x,s=(Mf,Nf))
    hh = np.rot90(h,2) # ribaltamento della maschera
    H = np.fft.fft2(hh,s=(Mf,Nf))
    Y = X*H
    P = P//2
    Q = Q//2
    y = np.real(np.fft.ifft2(Y))[P:(P+M),Q:(Q+N)]
    return y

y2 = filtrafreq(x,h)
MSE = np.mean((y1-y2)**2)
print('MSE = ', MSE)

plt.figure()
plt.subplot(1,3,1)
plt.imshow(x, clim=[0,255], cmap='gray')
plt.title('immagine')
plt.subplot(1,3,2)
plt.imshow(y1, clim=None, cmap='gray')
plt.title('filtraggio spaziale')
plt.subplot(1,3,3)
plt.imshow(y2, clim=None, cmap='gray')
plt.title('filtraggio in freq.')
