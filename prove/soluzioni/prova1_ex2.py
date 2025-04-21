# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 18:10:14 2025

prova 1, ex 2
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi
from skimage.color import hsv2rgb, rgb2hsv
plt.close('all')

x_ori = np.float64(io.imread('pears.png'))/255
x = np.float64(io.imread('pears_noise.png'))/255
plt.figure()
plt.imshow(x)
plt.title('immagine rumorosa')
x_hsv = rgb2hsv(x)
V = x_hsv[:,:,2]

plt.figure()
plt.imshow(V,clim=[0,1], cmap='gray')
plt.title("luminanza del'immagine rumorosa")

M,N = V.shape
fV = np.fft.fftshift(np.fft.fft2(V))
m = np.fft.fftshift(np.fft.fftfreq(M))
n = np.fft.fftshift(np.fft.fftfreq(N))
l,k = np.meshgrid(n,m)

plt.figure()
plt.imshow(np.log(1+np.abs(fV)),clim=None, cmap='gray', extent=(-0.5,+0.5,+0.5,-0.5))
plt.title("spettro di luminanza del'immagine rumorosa")

#from mpl_toolkits.mplot3d import Axes3D
#ax = Axes3D(plt.figure(), auto_add_to_figure=True) # crea una figura per i grafici 3d
#ax.plot_surface(l,k,np.log(1+np.abs(fV)), linewidth=0, cmap='jet')
#plt.title("spettro di luminanza del'immagine rumorosa")

H = np.ones((M,N), np.bool_)
list_point = [(-0.1,0.1),(-0.2,0.2),(-0.3,0.3),(-0.4,0.4),(-0.5,0.5)]
B = 0.015
B2 = 0.005
B3 = 0.1
for i in range(4):
    nu = list_point[i][1]
    mu = list_point[i][0]
    
    D1 = np.sqrt((k-mu)**2+(l-nu)**2)
    D2 = np.sqrt((k+mu)**2+(l+nu)**2)
    H = H & (D1>B) & (D2>B)
nu = list_point[0][1]
mu = list_point[0][0]
R1 = (np.abs(k-mu) > B2) | (np.abs(l-nu) > B3)
R2 = (np.abs(k+mu) > B2) | (np.abs(l+nu) > B3)
R3 = (np.abs(k-mu) > B3) | (np.abs(l-nu) > B2)
R4 = (np.abs(k+mu) > B3) | (np.abs(l+nu) > B2)
H = H & R1 & R2 & R3 & R4


plt.figure()
plt.imshow(H,clim=[0,1], cmap='gray', extent=(-0.5,+0.5,+0.5,-0.5))
plt.title("filtro in frequenza")

fV = fV * H

plt.figure()
plt.imshow(np.log(1+np.abs(fV)),clim=None, cmap='gray', extent=(-0.5,+0.5,+0.5,-0.5))
plt.title("spettro di luminanza del'immagine filtrata")

V = np.real(np.fft.ifft2(np.fft.ifftshift(fV)))

plt.figure()
plt.imshow(V,clim=[0,1], cmap='gray')
plt.title("luminanza del'immagine filtrata")

H = x_hsv[:,:,0]
S = x_hsv[:,:,1]
y_hsv = np.stack((H,S,V),2)
y = hsv2rgb(y_hsv)

MSE = np.mean((x_ori-y)**2)
print('MSE=', MSE)
#MSE= 9.089287001374027e-05
plt.figure()
plt.imshow(y)
plt.title("immagine filtrata")

S = S ** 0.6
y_hsv = np.stack((H,S,V),2)
y = hsv2rgb(y_hsv)

plt.figure()
plt.imshow(y)
plt.title("immagine filtrata con enhancement dei colori")
