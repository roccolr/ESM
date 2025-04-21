import numpy as np 
import matplotlib.pyplot as plt
import scipy.ndimage as ndi 
import skimage.io as io 

def dehaze(x):
    return np.log(x)
    
def custom_filter(x):
    th = 4
    if np.std(x) >= th:
        return np.median(x)
    else:
        return x[int(len(x)/2+1)]

def remove_noise(x):
    y = ndi.generic_filter(x, custom_filter, (3,3))
    y = ndi.gaussian_filter(y, 0.6)
    return y

def fshs(x,k):
    return (k-1)*(x-np.min(x))/(np.max(x)-np.min(x))


if __name__ == '__main__':
    path = 'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\esercitazioni\\simulazione_11_04_25\\esercitazione_250411\\immagini\\'

    im = path + 'ponte.png'
    x = np.float32(io.imread(im))
    y = remove_noise(x)
    y = fshs(y,256)
    y = y/256
    y = y**0.9

    plt.figure(1)
    plt.imshow(x, clim=[0,255], cmap='gray')
    plt.colorbar()
    plt.title('input')
    plt.figure(2)
    plt.imshow(y, clim=[0,1], cmap='gray')
    plt.colorbar()
    plt.title('output')
    plt.show()