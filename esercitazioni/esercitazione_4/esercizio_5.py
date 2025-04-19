import numpy as np 
import matplotlib.pyplot as plt
import skimage.io as io 
import scipy.ndimage as ndi 


if __name__ == '__main__':
    path = 'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\Immagini\\'
    im = path + 'car.tif'

    x = np.float32(io.imread(im))
    X = np.fft.fft2(x)

    # progetto filtro 
    M,N = x.shape
    m = np.fft.fftshift(np.fft.fftfreq(M))
    n = np.fft.fftshift(np.fft.fftfreq(N))
    l,k = np.meshgrid(n,m)
    mu = 0.18
    nu = 0.16
    B1 = 0.05
    W1 = 0.002
    D1 = (((k-mu)**2 + (l-nu)**2)**0.5) > B1
    D2 = (((k-mu)**2 + (l+nu)**2)**0.5) > B1
    D3 = (((k+mu)**2 + (l+nu)**2)**0.5) > B1
    D4 = (((k+mu)**2 + (l-nu)**2)**0.5) > B1
    R1 = (np.abs(k-mu) >= W1) & (np.abs(l-nu)>= W1)
    R2 = (np.abs(k+mu) >= W1) & (np.abs(l-nu)>= W1)
    R3 = (np.abs(k-mu) >= W1) & (np.abs(l+nu)>= W1)
    R4 = (np.abs(k+mu) >= W1) & (np.abs(l+nu)>= W1)
    H1 = (D1 & D2 & D3 & D4) & (R1 & R2 & R3 & R4)

    B2 = 0.02
    W2 = 0.001
    mu = 0.33
    nu = 0.16
    D5 = (((k-mu)**2 + (l-nu)**2)**0.5) > B2
    D6 = (((k-mu)**2 + (l+nu)**2)**0.5) > B2
    D7 = (((k+mu)**2 + (l+nu)**2)**0.5) > B2
    D8 = (((k+mu)**2 + (l-nu)**2)**0.5) > B2
    R5 = (np.abs(k-mu) >= W2) & (np.abs(l-nu)>= W2)
    R6 = (np.abs(k+mu) >= W2) & (np.abs(l-nu)>= W2)
    R7 = (np.abs(k-mu) >= W2) & (np.abs(l+nu)>= W2)
    R8 = (np.abs(k+mu) >= W2) & (np.abs(l+nu)>= W2)
    H2 = (D5 & D6 & D7 & D8) & (R5 & R6 & R7 & R8)

    H = H1 & H2

    Y = np.fft.fftshift(X)*H
    y = np.real(np.fft.ifft2(np.fft.fftshift(Y)))

    plt.close('all')
    plt.figure(1)
    plt.imshow(x, cmap='gray', clim=[0,255])
    plt.title('input')
    plt.colorbar()
    plt.figure(2)
    plt.imshow(np.log(1+np.abs(np.fft.fftshift(X))), clim=None, cmap='gray', extent=(-0.5, 0.5, 0.5, -0.5))
    plt.title('FFT input')
    plt.figure(3)
    plt.imshow(H, clim=None, cmap='gray', extent=(-0.5, 0.5, 0.5, -0.5))
    plt.title('FDT')
    plt.figure(4)
    plt.imshow(y, clim=None, cmap='gray')
    plt.title('output')
    plt.colorbar()
    plt.show()