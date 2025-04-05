import numpy as np 
import matplotlib.pyplot as plt
import scipy.ndimage as ndi 
import skimage.io as io
import sys
sys.path.insert(1,'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio')
from my_modules.histogramop import fshs
from my_modules.my_lib import *

path = 'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\Immagini\\'

def add_noise(x,d):
    M,N = x.shape
    n = d*np.random.randn(M,N)
    return x+n 

def space_adaptive_filter(x,d):
    global_var = np.var(x)
    local_var = ndi.generic_filter(x, np.var, (7,7))
    local_mean = ndi.generic_filter(x, np.mean, (7,7))

    y = x - (d**2)*(x-local_mean)/local_var
    return y



if __name__ == '__main__':
    im = path+'lena.jpg'
    x = np.float32(io.imread(im))
    

    l1 = []
    l2 = []
    mse_l1 = []
    mse_l2 = []
    for i in range(0,15):
        noisy = add_noise(x,15)
        y = space_adaptive_filter(noisy, 7)
        z = ndi.uniform_filter(noisy, size=7)
        l1.append(y)
        l2.append(z)

        mse_l1.append(np.mean((noisy-y)**2))
        mse_l2.append(np.mean((noisy-z)**2))


    plt.figure(1)
    plt.imshow(noisy, clim=[0,255], cmap='gray')
    plt.figure(2)
    plt.imshow(l1[5], clim=[0,255], cmap='gray')
    plt.figure(3)
    plt.imshow(l2[5], clim=[0,255], cmap='gray')

    plt.figure(4)
    x = range(0,15)
    y = mse_l1
    z = mse_l2
    plt.subplot(1,2,1)
    plt.plot(x,y)
    plt.title('mse adaptive')
    plt.subplot(1,2,2)
    plt.plot(x,z)
    plt.title('mse_mean')

    plt.show()