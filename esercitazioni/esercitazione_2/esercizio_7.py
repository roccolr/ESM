import numpy as np 
import matplotlib.pyplot as plt
import scipy.ndimage as ndi 
import skimage.io as io 
import skimage.data as dt
import skimage.transform as tf


def rot_shear(x, theta, c):
    # rotazione rispetto al centro e poi distorsione verticale
    M,N = np.shape(x)

    #rotazione rispetto al centro 
    Tt1 = np.array([[1,0,0],[0,1,0],[-M/2,-N/2,1]],dtype=np.float32)
    Tr = np.array([[np.cos(theta),np.sin(theta),0],[-np.sin(theta),np.cos(theta),0],[0,0,1]], dtype=np.float32)
    Tt2 = np.array([[1,0,0],[0,1,0],[M/2,N/2,1]],dtype=np.float32)

    Tr = np.dot((np.dot(Tt1, Tr)), Tt2)

    #distorsione
    Td = np.array([[1,0,0], [c,1,0], [0,0,1]], dtype=np.float32)
    M = np.dot(Tr, Td)
    A = M[[1,0,2],:][:,[1,0,2]].T

    y = tf.warp(x, A, cval=0)
    return y

if __name__=='__main__':
    x = np.float64(dt.checkerboard())
    y = rot_shear(x, np.pi/6, -0.5)

    # distorsione rispetto al centro?

    plt.figure(1)
    plt.subplot(1,2,1)
    plt.imshow(x)
    plt.title('originale')
    plt.subplot(1,2,2)
    plt.imshow(y)
    plt.title('modificata')
    plt.show()