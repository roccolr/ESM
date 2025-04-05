import numpy as np 
import matplotlib.pyplot as plt
import scipy.ndimage as ndi 
import skimage.io as io
import sys
sys.path.insert(1,'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio')
from my_modules.histogramop import fshs
from my_modules.my_lib import *

def smooth(x):
    h = np.array([[1,2,1], [2,4,2], [1,2,1]])/16
    y = ndi.correlate(x, h, mode='reflect')
    return y

path = 'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\Immagini\\'

if __name__ == '__main__':
    im = path + 'lena.jpg'
    x = np.float32(io.imread(im))
    y = smooth(x)

    plt.close('all')
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.imshow(x, clim=[0,255], cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(y, clim=[0,255], cmap='gray')
    plt.show()