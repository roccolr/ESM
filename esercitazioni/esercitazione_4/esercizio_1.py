"""
Spettro di ampiezza. Analizzate lo spettro di ampiezza di alcune immagini di test (circuito.jpg, impronta.tif, anelli.tif), siete in grado di legare il contenuto in frequenza con l’andamento spaziale dell’immagine?
"""

import numpy as np 
import skimage.io as io 
import skimage.util as ut
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
path = 'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\Immagini\\'

if __name__ == '__main__':
    im1 = path+'circuito.jpg'
    im2 = path+'impronta1.tif'
    im3 = path+'anelli.tif'

    x1 = np.float32(io.imread(im1)) # circuito
    x2 = np.float32(io.imread(im2)) # impronta
    x3 = np.float32(io.imread(im3)) # anelli

    X1 = np.fft.fft2(x1)
    X2= np.fft.fft2(x2)
    X3 = np.fft.fft2(x3)

    plt.close('all')
    plt.figure(1)
    plt.imshow(x1, clim=[0,255], cmap='gray')
    plt.colorbar()
    plt.title('circuito.jpg')
    plt.figure(2)
    plt.imshow(x2, clim=[0,255], cmap='gray')
    plt.colorbar()
    plt.title('impronta.tif')
    plt.figure(3)
    plt.imshow(x3, clim=[0,255], cmap='gray')
    plt.colorbar()
    plt.title('anelli.tif')
    plt.figure(4)
    plt.imshow(np.log(1+np.abs(np.fft.fftshift(X1))), clim=None, cmap='jet', extent=(-0.5,0.5,0.5,-0.5))
    plt.title('fft(circuito)')
    plt.colorbar()
    plt.figure(5)
    plt.imshow(np.log(1+np.abs(np.fft.fftshift(X2))), clim=None, cmap='jet', extent=(-0.5,0.5,0.5,-0.5))
    plt.title('fft(impronta)')
    plt.colorbar()
    plt.figure(6)
    plt.imshow(np.log(1+np.abs(np.fft.fftshift(X1))), clim=None, cmap='jet', extent=(-0.5,0.5,0.5,-0.5))
    plt.title('fft(anelli)')
    plt.colorbar()
    plt.show()


