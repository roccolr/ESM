import numpy as np
import matplotlib.pyplot as plt 
import skimage.io as io 
import skimage.morphology as moprh
import scipy.ndimage as ndi 


if __name__ == '__main__':
    path = 'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\Immagini\\'
    im = path + 'testo_fax.tif'

    x = np.float32(io.imread(im))

    plt.figure(1)
    plt.imshow(x, clim=[0,1], cmap='gray')

    # definiamo la croce

    b = np.array([[0,1,0],[1,1,1],[0,1,0]], np.bool)
    y = moprh.binary_dilation(x, b)

    plt.figure(2)
    plt.imshow(y, clim=[0,1], cmap='gray')

    plt.show()

