import numpy as np 
import scipy.ndimage as ndi 
import matplotlib.pyplot as plt
import skimage.io as io 
import skimage.color as clr


if __name__ == '__main__':
    path = 'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\Immagini\\'
    im1 = path + 'fragole.jpg'
    im2 = path + 'cubo.jpg'
    x1 = np.float32(io.imread(im1))
    x2 = np.float32(io.imread(im2))
    x1 = x1/np.max(x1)
    x2 = x2/np.max(x2)

    hsv_x1 = clr.rgb2hsv(x1)
    hsv_x2 = clr.rgb2hsv(x2)

    plt.figure(1)
    plt.imshow(x1, clim=[0,1])
    plt.title('fragole')
    plt.figure(2)
    plt.imshow(x2, clim=[0,1])
    plt.title('cubo')
    plt.figure(3)
    plt.subplot(1,3,1)
    plt.imshow(hsv_x1[:,:,0], clim=[0,1], cmap='gray')
    plt.title('fragole_h')
    plt.subplot(1,3,2)
    plt.imshow(hsv_x1[:,:,1], clim=[0,1], cmap='gray')
    plt.title('fragole_s')
    plt.imshow(hsv_x1[:,:,2], clim=[0,1], cmap='gray')
    plt.subplot(1,3,3)
    plt.title('fragole_v')
    plt.figure(4)
    plt.subplot(1,3,1)
    plt.imshow(hsv_x2[:,:,0], clim=[0,1], cmap='gray')
    plt.title('cubo_h')
    plt.subplot(1,3,2)
    plt.imshow(hsv_x2[:,:,1], clim=[0,1], cmap='gray')
    plt.title('cubo_s')
    plt.subplot(1,3,3)
    plt.imshow(hsv_x2[:,:,2], clim=[0,1], cmap='gray')
    plt.title('cubo_v')
    plt.show()