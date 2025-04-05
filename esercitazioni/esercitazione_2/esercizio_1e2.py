# realizzare enhancement globale dell'immagine con equalizzazione dell'istogramma, scrivendo la funzione
# y = glob_equaliz(x) e mostrare l'immagine elaborata

import numpy as np 
import matplotlib.pyplot as plt
import skimage.io as io 
import scipy.ndimage as ndi
import sys
sys.path.insert(1,'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio')
import my_modules.histogramop as hope
import my_modules.my_lib as my


def local_equalize(x):
    g_avg = np.median(x)
    l_avg = ndi.generic_filter(x, np.mean, (3,3))
    m = l_avg < g_avg*0.2
    y = 50*m*x + (1-m)*x
    
    return y,m

def glob_equalize(x):
    y = x ** 0.25
    y = hope.fshs(y,256)
    return y
    

if __name__ == '__main__':
    path = 'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\immagini\\Immagini\\quadrato.tif'
    x = np.float64(io.imread(path))
    # y = glob_equalize(x)
    y,m = local_equalize(x)

    # print
    plt.figure(1)
    plt.subplot(2,2,1)
    plt.imshow(x, clim=[0,255], cmap='gray')
    plt.subplot(2,2,2)
    plt.hist(x.flatten(), bins=256)
    plt.subplot(2,2,3)
    plt.imshow(y, clim=[0,255], cmap='gray')
    plt.subplot(2,2,4)
    plt.hist(y.flatten(), bins=256)

    plt.figure(2)
    plt.imshow(m, clim=None, cmap='gray')
    plt.show()