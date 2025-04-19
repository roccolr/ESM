import matplotlib.pyplot as plt 
import numpy as np 
import scipy.ndimage as ndi 
import skimage.io as io 

path = 'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\Immagini\\'

if __name__ == '__main__':
    im = path + 'anelli.tif'
    x = np.float32(io.imread(im))
    X = np.fft.fft2(x)

    M,N = x.shape
    m = np.fft.fftshift(np.fft.fftfreq(M))
    n = np.fft.fftshift(np.fft.fftfreq(N))
    l,k = np.meshgrid(n,m)
    R = (np.abs(k) >= 0.003) & (np.abs(l)>=0.003)
    D = (k**2 + l**2)**0.5
    C = D < 0.02
    H = R | (C)
    Y = np.fft.fftshift(X)*H
    y = np.real(np.fft.ifft2(np.fft.fftshift(Y)))

    plt.figure(1)
    plt.imshow(x, clim=None, cmap='gray')
    plt.title('input')
    plt.figure(2)
    plt.imshow(np.log(np.abs(np.fft.fftshift(X))+1), clim=None, cmap='gray', extent=(-0.5,0.5,0.5,-0.5))
    plt.title('FFT INPUT')
    plt.figure(3)
    plt.imshow(H, clim=None, cmap='gray',  extent=(-0.5,0.5,0.5,-0.5))
    plt.title('fdt')
    plt.figure(4)
    plt.imshow(y, clim=None, cmap='gray')
    plt.title('output')
    plt.show()
