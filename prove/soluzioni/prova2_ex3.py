# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 17:56:22 2025

prova 2, ex 3
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi
from skimage.feature import canny
import skimage.morphology as morph
plt.close('all')

x = np.reshape(np.fromfile('target_rumorosa.raw', np.float32), (256,256))

plt.figure()
plt.imshow(x,clim=[0,255], cmap='gray')
plt.title('immagine')

## strategia 1 standard
mappa1 = canny(x, sigma=2.0, low_threshold=0.5, high_threshold=0.78, use_quantiles=True)
plt.figure()
plt.imshow(mappa1,clim=[0,1], cmap='gray')
plt.title('img strategia standard')


## strategia 2
# ho scelto K=5; T=1.2
def block_fun(x):
    y = np.mean(x) / ( np.prod(x) ** (1/len(x)))
    return y

k = 5
T = 1.15
R_AG = ndi.generic_filter(x**2, block_fun, (k,k))
mappa2 = R_AG>=T 

plt.figure()
plt.imshow(mappa2, clim=[0,1], cmap='gray')
plt.title('mappa strategia 2')

mappa = morph.opening(mappa2, morph.disk(1))
mappa = morph.thin(mappa)
plt.figure()
plt.imshow(mappa, clim=[0,1], cmap='gray')
plt.title('mappa finale')


"""
def block_fun(x, r):
    xc = x[x.shape[0]//2] # valore al centro del blocco
    y = x[np.abs(x-xc)<=r] # elementi nel range
    if len(y)<4:
        v = np.mean(x)
    else:
        v = np.mean(y)
    return v
    
def filtro_sigma(x, K, sigma):
    y = ndi.generic_filter(x, block_fun, (K,K), extra_arguments=(2*sigma,))
    return y

x = np.float64(io.imread('barbara.png'))
M,N = x.shape
sigma = 20
noise = sigma * np.random.randn(M,N)
noisy = x + noise

K = 7
y = filtro_sigma(noisy, K, sigma)

MSE = np.mean((x-y)**2)
PSNR = 10*np.log10(255**2/MSE)
print('PSNR=', PSNR)

plt.figure(1)
plt.subplot(1,3,1)
plt.imshow(x, clim=[0,255], cmap='gray')
plt.title('originale')
plt.subplot(1,3,2)
plt.imshow(noisy, clim=[0,255], cmap='gray')
plt.title('rumorosa')
plt.subplot(1,3,3)
plt.imshow(y, clim=[0,255], cmap='gray')
plt.title('filtrata')


"""

"""

% mappa
figure(3);imshow(mappa2,[]);title('mappa strategia 2');
str=ones(3);
mappaopp=imopen((mappa2),str);
figure(4);imshow(mappaopp,[]);title('mappa op morf 1');

mappafin=bwmorph(mappaopp,'thin',2);
figure(5);imshow(mappafin,[]);title('mappa finale strategia 2 ');
"""