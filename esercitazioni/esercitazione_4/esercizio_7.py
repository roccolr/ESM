import numpy as np 
import matplotlib.pyplot as plt
import skimage.io as io 
import scipy.ndimage as ndi 
from skimage.color import rgb2gray

def detect(x):
    tau = 0.35
    x = rgb2gray(x)/np.max(x)
    X = np.fft.fft2(x)
    H = np.log(np.abs(X))

    M,N = x.shape
    m = np.fft.fftfreq(M)
    n = np.fft.fftfreq(N)
    l,k = np.meshgrid(n,m)

    mask = (np.abs(k)<=tau) & (np.abs(l)<=tau)
    d = np.sum(H[mask])/np.sum(H)
    return d

if __name__ == '__main__':
    th = 0.7 
    path = 'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\Immagini\\'

    pics = ['volto1.png', 'volto2.png', 'volto3.png', 'volto4.png']

    for pic in pics:
        x = np.float32(io.imread(path+pic))/255
        plt.figure()
        plt.imshow(x, clim=None)
        if detect(x) < 0.7:
            plt.title('sintetica')
        else:
            plt.title('originale')
        
    plt.show()