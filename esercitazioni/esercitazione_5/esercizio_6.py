import numpy as np 
import matplotlib.pyplot as plt 
import scipy.ndimage as ndi 
import skimage.io as io 
from sklearn.cluster import k_means

def thresholding_locale(x):
    a = 30
    b = 1.5
    avg_glob = np.mean(x)
    std_loc = ndi.generic_filter(x, np.std, (3,3))
    map = (x>std_loc*a) & (x>avg_glob*b)
    return map

if __name__ == '__main__':
    path = 'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\Immagini\\'
    im = path + 'yeast.tif'

    x = np.float32(io.imread(im))
    map = thresholding_locale(x)

    plt.figure(1)
    plt.imshow(x, clim=[0,255], cmap='gray')
    plt.title('input')
    plt.colorbar()
    plt.figure(2)
    plt.imshow(map, clim=[0,1], cmap='gray')
    plt.title('label map')
    
    plt.show()