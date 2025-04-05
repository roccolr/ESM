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
    noise = d*np.random.randn(M,N)
    return x+noise

def denoise_mean(x, k:int):
    return ndi.uniform_filter(x, size=k)

def denoise_gaussian(x, d):
    return ndi.gaussian_filter(x, sigma=d)

def mse(x,y):
    return np.mean((x-y)**2)

def denoise_custom(x):
    h = np.array([[1,2,1], [2,4,2], [1,2,1]])/16
    y = ndi.correlate(x, h, mode='reflect')
    return y

if __name__ == '__main__':
    im = path+'lena.jpg'

    x = np.float32(io.imread(im))
    noisy = add_noise(x, 10)

    # effettuiamo lo smoothing con media, sfocatura gaussiana e filtro esercizio_2

    y1 = denoise_mean(noisy, 3)
    y2 = denoise_gaussian(noisy, 2)
    y3 = denoise_custom(noisy)

    plt.close('all')

    plt.figure(1)
    plt.subplot(1,2,1)
    plt.imshow(x, clim=[0,255], cmap='gray')
    plt.title('originale')
    plt.subplot(1,2,2)
    plt.imshow(noisy, clim=[0,255], cmap='gray')
    plt.title('noisy')
    
    plt.figure(2)
    plt.subplot(1,3,1)
    plt.imshow(y1, clim=[0,255], cmap='gray')
    plt.subplot(1,3,2)
    plt.imshow(y2, clim=[0,255], cmap='gray')
    plt.subplot(1,3,3)
    plt.imshow(y3, clim=[0,255], cmap='gray')
    plt.show()

    print("SCARTI QUADRATICI MEDI:")
    print(f"y1\t{mse(x,y1)}")
    print(f"y2\t{mse(x,y2)}")
    print(f"y3\t{mse(x,y3)}")
    