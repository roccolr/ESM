import numpy as np 
import matplotlib.pyplot as plt 
import scipy.ndimage as ndi
import skimage.io as io 

path = "C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\esercitazioni\\simulazione_11_04_25\\esercitazione_250411\\ponte.png"

def fshs(x, k:int):
    x_min = np.min(x)
    x_max = np.max(x)
    return ((k-1)*(x-x_min)/(x_max-x_min))

def sp_filter(x):
    return ndi.median_filter(x, size=(3,3), mode='reflect')

# path = 'ponte.gif'
if __name__ == '__main__':
    x = np.float32(io.imread(path))
    # y = sp_filter(x)
    # y = y**1.3
    x[x<115] = 115
    x[x>135] = 135 
    y = fshs(x, 256)
    y = sp_filter(y)

    plt.close('all')
    plt.figure(1)
    plt.imshow(x, clim=[0,255], cmap='gray')
    plt.colorbar()
    plt.figure(2)
    plt.hist(x.flatten(), bins=256)
    plt.figure(3)
    plt.imshow(y, clim=[0,255], cmap='gray')
    plt.colorbar()
    plt.figure(4)
    plt.hist(y.flatten(), bins=256)
    plt.show()