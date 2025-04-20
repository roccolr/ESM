import numpy as np 
import matplotlib.pyplot as plt
import skimage.io as io 
import scipy.ndimage as ndi 
from skimage.transform import warp

if __name__ == '__main__':
    path = 'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\Immagini\\'
    im = path+'mare.png'
    x = np.float32(io.imread(im))

    # punto A
    X = np.fft.fft2(x)
    R = np.real(np.fft.ifft2(np.abs(X)**2))
    R = np.fft.fftshift(R)
    Nr, Nc = R.shape
    m = np.arange(Nr) - Nr/2
    n = np.arange(Nc) - Nc/2
    l,k = np.meshgrid(n,m)

    # punto b
    L = ndi.generic_filter(R, np.max, (5,5)) == R
    p = np.sort(R[L])
    map = L & (R == p[-2])

    #punto c
    tm = k[map]
    tn = l[map]

    A = np.array([[1,0, tn[0]], [0, 1, tm[0]], [0, 0, 1]])
    y = warp(x, A)
    mask = x==y

    plt.figure(1)
    plt.imshow(x, clim=None, cmap='gray')
    plt.title('input')
    plt.colorbar()
    plt.figure(2)
    plt.imshow(R, clim=None, cmap='gray', extent=(-Nc/2, Nc/2-1, Nr/2-1, -Nr/2))
    plt.title('autocorrelazione')
    plt.colorbar()
    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(k,l,R,linewidth=0, cmap='jet')

    plt.figure(4)
    plt.imshow(mask, clim=0, cmap='gray')

    plt.show()