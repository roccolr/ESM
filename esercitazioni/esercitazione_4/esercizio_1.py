import numpy as np 
import skimage.io as io 
import skimage.util as ut
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
path = 'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\Immagini\\'

if __name__ == '__main__':
    # im = path+'circuito.jpg'
    # im = path+'impronta2.tif'
    im = path+'anelli.tif'

    x = np.float32(io.imread(im,plugin='tifffile'))
    X = np.fft.fft2(x)
    Y = np.log(1+np.abs(np.fft.fftshift(X)))

    plt.figure(1)
    plt.imshow(x, clim=[0,255], cmap='gray')
    plt.title('originale')
    plt.colorbar()
    plt.figure(2)
    plt.imshow(Y, clim=None, cmap='gray', extent=(-0.5,+0.5,+0.5,-0.5))
    plt.title('fft')
    plt.colorbar()
    plt.show()
