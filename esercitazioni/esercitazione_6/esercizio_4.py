import numpy as np 
import matplotlib.pyplot as plt 
import skimage.io as io 
import skimage.morphology as morph
import scipy.ndimage as ndi 
import skimage.color as clr
from sklearn.cluster import k_means

'''
Morfologia: La useremo come operazione dopo la segmentazione. Cerchiamo di separare le cellule chiare dalle cellule scure, producendo due mappe pulite in output.
'''

if __name__ == '__main__':
    path = 'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\Immagini\\'
    im = path + 'cells.png'

    x = np.float32(io.imread(im))
    x = clr.rgb2gray(x[:,:,:-1])
    M,N = x.shape
    centroid, idx, sum_var = k_means(np.reshape(x, (M*N,1)), 3)

    dark_cells = (np.reshape(idx, (x.shape)) == 1)
    s = morph.footprint_rectangle((7,7))
    dark_cells = morph.binary_opening(dark_cells, s)

    plt.figure(1)
    plt.imshow(x, clim=[0,255], cmap='gray')
    plt.title('input')
    plt.colorbar()
    plt.figure(2)
    plt.imshow(np.reshape(idx, (x.shape)), clim=None, cmap='jet')
    plt.title('clusters')
    plt.figure(3)
    plt.imshow(dark_cells, clim=None, cmap='gray')
    plt.title('clusters')

    plt.show()