import numpy as np 
import skimage.io as io 
import skimage.util as ut
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import sys
sys.path.append('C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio')
from my_modules.seg_utils import zero_crossing
path = 'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\Immagini\\'



if __name__ == '__main__':
    x = np.float32(io.imread(path+'luna.jpg'))
    mat = [[0,0,1,0,0], [0,0,0,0,0], [1,0,-4,0,1], [0,0,0,0,0],[0,0,1,0,0]]
    h = np.array(mat, dtype=np.float32)
    y = ndi.correlate(x,h, mode='reflect')
    z = x - y
    map = zero_crossing(y)

    plt.close('all')
    plt.figure(1)
    plt.imshow(x, clim=[0,255], cmap='gray')
    plt.colorbar()
    plt.title('input')

    plt.figure(2)
    plt.imshow(y, clim=None, cmap='gray')
    plt.colorbar()
    plt.title('corr. result')

    plt.figure(3)
    plt.imshow(z, clim=None, cmap='gray')
    plt.title('enhanced')

    plt.figure(4)
    plt.imshow(map, clim=None, cmap='gray')
    plt.title('map')
    plt.show()
