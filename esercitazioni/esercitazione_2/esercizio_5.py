# Scrivere la funzione che realizzi la distorsione di un'immagine

import numpy as np 
import matplotlib.pyplot as plt
import scipy.ndimage as ndi 
import skimage.io as io 
from skimage.transform import warp 
from skimage.transform import rescale


def deforma(x:np.ndarray, c:float, d:float)->np.ndarray:
    """c verticale, d orizzontale"""
    # deformazione orizzontale
    # x.shape = (M, N+d) c'è direttamente output_shape
    M,N = x.shape
    T = np.array([[1,d,0], [0,1,0], [0,0,1]])
    A = T[[1,0,2],:][:,[1,0,2]].T
    y = warp(x, A, order=1, output_shape=(M,N))

    # deformazione verticale
    M,N = y.shape
    T = np.array([[1,0,0], [c, 1, 0], [0,0,1]])
    A = T[[1,0,2],:][:,[1,0,2]].T
    z = warp(y, A,order=1, output_shape=(M,N))

    return z



if __name__=='__main__':
    path = 'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\immagini\\Immagini\\lena.jpg'
    x = np.float64(io.imread(path))
    # x = np.reshape(x, (x.shape[0]+1, x.shape[1]+1))

    y = deforma(x,1.1,0)

    plt.figure(1)
    plt.subplot(1,2,1)
    plt.imshow(x, clim=[0,255], cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(y, clim=[0,255], cmap='gray')
    plt.show()