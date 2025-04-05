import numpy as np 
import matplotlib.pyplot as plt
import scipy.ndimage as ndi 
import skimage.io as io
import sys
sys.path.insert(1,'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio')
from my_modules.histogramop import fshs
from my_modules.my_lib import *

path = 'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\Immagini\\'

if __name__ == '__main__':
    im = path+'space.jpg'
    x = np.float64(io.imread(im))

    # 1 - Visualizza l'immagine 
    plt.figure(1)
    plt.imshow(x, clim=[0,255], cmap='gray')


    # 2 - Calcolo media aritmetica 
    k = 15                                  #dimensione del kernel
    h = np.ones((k,k))/(k**2)               # creazione maschera
    y = ndi.correlate(x,h,mode='reflect')

    plt.figure(2)
    plt.imshow(y, clim=[0,255], cmap='gray')

    # 3 - operazione di thresholding
    max_im = np.max(y)
    tsh = 0.25 * max_im 

    mask = y<tsh
    y = 0*x*mask+(1-mask)*x

    '''
    ATTENZIONE!!!
    La maschera va applicata all'immagine originale, l'immagine filtrata serve solo per definire la maschera
    '''

    plt.figure(3)
    plt.imshow(y, clim=[0,255], cmap='gray')
    plt.show()