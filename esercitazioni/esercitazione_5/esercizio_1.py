import numpy as np 
import matplotlib.pyplot as plt
import skimage.io as io 
import skimage.color as clr 
import scipy.ndimage as ndi 

def rgb2cmy(x):
    R,G,B = (x[:,:,0], x[:,:,1], x[:,:,2])
    C,M,Y = (1-R, 1-G, 1-B)
    y = np.stack((C,M,Y), -1)
    return y

def easy_rgb2cmy(x):
    return 1.0 - x

def rgb2cmyk(x):
    z = 1.0 - x
    K = np.min(z, axis=-1)
    C,M,Y = (z[:,:,0]-K, z[:,:,1]-K, z[:,:,2]-K)
    return np.stack((C,M,Y,K), -1)

if __name__ == '__main__':
    path = 'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\Immagini\\'
    im = path + 'lenac.jpg'
    x = np.float32(io.imread(im))
    x = x/np.max(x)

    # y1 = rgb2cmy(x)
    y1 = easy_rgb2cmy(x)
    y2 = rgb2cmyk(x)

    plt.figure(1)
    plt.imshow(x, clim=[0,1])
    plt.title('input')
    plt.figure(2)
    plt.subplot(1,3,1)
    plt.imshow(y1[:,:,0], clim=[0,1], cmap='gray')
    plt.title('Ciano')
    plt.subplot(1,3,2)
    plt.imshow(y1[:,:,1], clim=[0,1], cmap='gray')
    plt.title('Magenta')
    plt.subplot(1,3,3)
    plt.imshow(y1[:,:,2], clim=[0,1], cmap='gray')
    plt.title('Giallo')
    plt.figure(3)
    plt.subplot(1,4,1)
    plt.imshow(y2[:,:,0], clim=[0,1], cmap='gray')
    plt.title('Ciano')
    plt.subplot(1,4,2)
    plt.imshow(y2[:,:,1], clim=[0,1], cmap='gray')
    plt.title('Magenta')
    plt.subplot(1,4,3)
    plt.imshow(y2[:,:,2], clim=[0,1], cmap='gray')
    plt.title('Giallo')
    plt.subplot(1,4,4)
    plt.imshow(y2[:,:,3], clim=[0,1], cmap='gray')
    plt.title('Nero')

    plt.show()

