'''
Applicare rumore sale e pepe ad un'immagine
'''

import numpy as np 
import skimage.io as io 
import skimage.util as ut
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

path = 'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\Immagini\\'

def add_noise(x):
    return ut.random_noise(x, mode = 's&p')

def median_filter(x, size:int):
    return ndi.generic_filter(x, np.median, (size,size))

def mse(x,y):
    return np.mean((x-y)**2)

if __name__ == '__main__':
    im = path + 'lena.jpg'
    x = np.float32(io.imread(im))/255   
    y = add_noise(x)
    z = median_filter(y, 5)

    data = [(5,mse(x,median_filter(y, 5))), (7,mse(x,median_filter(y, 9))), (9, mse(x,median_filter(y, 9)))]

    plt.close('all')
    plt.figure(1)
    plt.imshow(x, clim=[0,1], cmap='gray')
    plt.title('originale')
    plt.figure(2)
    plt.imshow(y, clim=None, cmap='gray')
    plt.title('rumorosa')
    plt.figure(3)
    plt.imshow(z, clim=[0,1], cmap='gray')
    plt.figure(4)

    asc = [5,7,9]
    ord = [data[0][1], data[1][1], data[2][1]]
    plt.plot(asc,ord)    
    plt.show()

    