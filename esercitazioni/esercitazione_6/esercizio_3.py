import numpy as np 
import matplotlib.pyplot as plt 
import skimage.io as io 
import skimage.morphology as morph
import scipy.ndimage as ndi 


if __name__ == '__main__':
    path = 'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\Immagini\\'
    im = path + 'impronta_bool.tif'
    x = io.imread(im)

    s = morph.footprint_rectangle((3,3))
    y = morph.binary_opening(x,s)
    z = morph.binary_closing(y,s)


    plt.figure(1)
    plt.imshow(x, clim=[0,1], cmap='gray')
    plt.title('input')
    plt.figure(2)
    plt.imshow(y, clim=[0,1], cmap='gray')
    plt.title('output1')
    plt.figure(3)
    plt.imshow(morph.thin(z,3), clim=[0,1], cmap='gray')
    plt.title('output2')
    plt.show()