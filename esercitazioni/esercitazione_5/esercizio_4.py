import numpy as np 
import matplotlib.pyplot as plt 
import scipy.ndimage as ndi 
import skimage.io as io 
import skimage.color as clr
import skimage.exposure as exp

def enhancement_globale(x,sigma):
    R,G,B = (x[:,:,0], x[:,:,1], x[:,:,2])
    return np.stack((exp.equalize_hist(R**sigma), exp.equalize_hist(G**sigma), exp.equalize_hist(B**sigma)),-2)

def dehazing(x):
    R,G,B = (x[:,:,0], x[:,:,1], x[:,:,2])
    A1 = 0.7020
    A2 = 0.7020
    A3 = 0.7098
    t0 = 0.1

    R1,G1,B1 = R/A1, G/A2, B/A3 
    k = np.min(np.stack((R1,B1,G1),-1), -1)
    x_dark = ndi.generic_filter(k, np.min, (15,15))
    t = 1 -0.95*x_dark

    map = t<t0 
    t[map] = 0
    den = t + t0
    
    # R
    y_R = (R - A1)/den + A1

    # G
    y_G = (G - A2)/den + A2

    # B
    y_B = (B - A3)/den + A3

    y = np.stack((y_R, y_G, y_B),-1)
    return exp.equalize_hist(y/np.max(y))


if __name__ == '__main__':
    path = 'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\Immagini\\'
    im = path + 'paesaggio.jpg'
    x = np.float32(io.imread(im))
    x = x/np.max(x)

    y1 = dehazing(x)
    y2 = enhancement_globale(x,0.2)

    plt.figure(1)
    plt.imshow(x, clim=None)
    plt.title('input')
    plt.figure(2)
    plt.imshow(y1, clim=None)
    plt.title('output1')
    plt.figure(3)
    plt.imshow(y1, clim=None)
    plt.title('output2')

    plt.show()