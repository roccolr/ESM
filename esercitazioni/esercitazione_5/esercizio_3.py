import numpy as np 
import matplotlib.pyplot as plt 
import scipy.ndimage as ndi 
import skimage.io as io 
import skimage.color as clr


if __name__ == '__main__':
    path = 'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\Immagini\\'
    im = path + 'foto_originale.tif'
    x = np.float32(io.imread(im))
    x = x/np.max(x)

    # passaggio in hsv
    y = clr.rgb2hsv(x)
    i = y[:,:,-1]

    # trasformata di fourier intensità
    I = np.fft.fftshift(np.fft.fft2(i))
    M,N = I.shape
    m = np.fft.fftshift(np.fft.fftfreq(M))
    n = np.fft.fftshift(np.fft.fftfreq(N))
    l,k = np.meshgrid(n,m)

    H = (np.abs(l)<=0.10) & (np.abs(k)<=0.25)
    Z = I*H

    z = np.real(np.fft.ifft2(np.fft.ifftshift(Z)))
    y[:,:,2] = z 
    y = clr.hsv2rgb(y)

    plt.figure(1)
    plt.imshow(x, clim=None)
    plt.title('input')
    plt.figure(2)
    plt.imshow(y, clim=None)
    plt.title('output')
    plt.show()
    