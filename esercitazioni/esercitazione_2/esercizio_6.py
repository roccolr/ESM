# rotazione centrale
import numpy as np 
import matplotlib.pyplot as plt
import scipy.ndimage as ndi 
import skimage.io as io 
from skimage.transform import warp 
from skimage.transform import rescale


def rotate(x, theta):
    M,N = np.shape(x)
    
    # sposto al centro
    Tt1 = np.array([[1,0,0],[0,1,0],[-M/2,-N/2,1]],dtype=np.float32)

    # rotazione
    Tr = np.array([[np.cos(theta),np.sin(theta),0],[-np.sin(theta),np.cos(theta),0],[0,0,1]], dtype=np.float32)

    #torno alla pos iniziale
    Tt2 = np.array([[1,0,0],[0,1,0],[M/2,N/2,1]],dtype=np.float32)

    M = np.dot((np.dot(Tt1, Tr)), Tt2)
    A = M[[1,0,2],:][:,[1,0,2]].T

    y = warp(x, A, order=1, cval=0)
    return y

if __name__=='__main__':
    path = 'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\immagini\\Immagini\\lena.jpg'
    x = np.float64(io.imread(path))
    y = rotate(x,(-np.pi/3))
    plt.close('all')
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.imshow(x, clim=[0,255], cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(y, clim=[0,255], cmap='gray')
    plt.show()