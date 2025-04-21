import numpy as np
import matplotlib.pyplot as plt 
import skimage.io as io 
import skimage.morphology as morph
import scipy.ndimage as ndi 


if __name__ == '__main__':
    path = 'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\Immagini\\'
    im = path + 'circbw_bool.tif'

    x = np.float32(io.imread(im))

    plt.figure(1)
    plt.imshow(x, clim=[0,1], cmap='gray')

    # definiamo l'elemento strutturante

    s = morph.footprint_rectangle((45,45))
    y = morph.binary_opening(x,s)

    plt.figure(2)
    plt.imshow(y, clim=[0,1], cmap='gray')

    plt.show()

