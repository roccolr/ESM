import numpy as np 
import matplotlib.pyplot as plt
import scipy.ndimage as ndi 
import skimage.io as io
import sys
sys.path.insert(1,'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio')
from my_modules.histogramop import fshs
from my_modules.my_lib import *

path = 'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\Immagini\\'

def add_gaussian_noise(x,d):
    N,M = x.shape
    noise = d*np.random.randn(M,N)
    return x+noise

if __name__=='__main__':
    # strategia di filtraggio di tipo adattivo

    im = path+'barbara.png'
    x = np.float32(io.imread(im))

    # visualizzazione input
    plt.close('all')
    plt.figure()
    plt.imshow(x, clim=[0,255], cmap='gray')
    plt.title('input')
    plt.colorbar()

    d = 5

    # aggiunta rumore
    noisy_x = add_gaussian_noise(x, d)
    plt.figure(2)
    plt.imshow(noisy_x, clim=None, cmap='gray')
    plt.title('noisy_input')
    plt.colorbar()

    # calcolo varianza globale dell'immagine 
    glob_var = np.var(noisy_x)

    #calcolo varianza locale dell'immagine
    local_var = ndi.generic_filter(noisy_x, np.var, (7,7))
    local_mean = ndi.generic_filter(noisy_x, np.mean, (7,7))

    y = noisy_x - (d**2)*(noisy_x-local_mean)/local_var 

    plt.figure(3)
    plt.imshow(y, clim=None, cmap='gray')
    plt.title('output')
    plt.colorbar()
    
    plt.show()