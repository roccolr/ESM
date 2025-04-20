import numpy as np 
import matplotlib.pyplot as plt 
import scipy.ndimage as ndi 
import skimage.io as io 

def elabora(x, r1, r2):
    X = np.fft.fftshift((np.fft.fft2(x)))

    M,N = X.shape
    m = np.fft.fftshift(np.fft.fftfreq(M))
    n = np.fft.fftshift(np.fft.fftfreq(N))
    l,k = np.meshgrid(n,m)

    H = ((np.abs(k)<=r2) & (np.abs(k)>= r1)) & ((np.abs(k)<=r2) & (np.abs(k)>= r1))
    Y = X*H
    omega = np.int8(Y[Y!=0])
    E = np.sum(np.abs(Y[Y!=0])**2)/np.sum(omega)
    return E

if __name__ == '__main__':
    path = 'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\Immagini\\'
    im1 = path+'impronta1.tif'
    im2 = path+'impronta2.tif'

    x1 = np.float32(io.imread(im1))
    x2 = np.float32(io.imread(im2))

    E1 = np.sum(np.abs(x1)**2)
    E2 = np.sum(np.abs(x2)**2)

    r1, r2 = (0.10, 0.25)
    EM1 = elabora(x1, r1, r2)
    EM2 = elabora(x2, r1, r2)

    d1 = EM1/E1 
    d2 = EM2/E2

    if d1 > d2:
        print('impronta 1 vera')
    else: 
        print('impronta 2 vera')

    plt.figure(1)
    plt.subplot(1,2,1)
    plt.imshow(x1, clim=[0,255], cmap='gray')
    plt.title('impronta 1')
    plt.subplot(1,2,2)
    plt.imshow(x2, clim=[0,255], cmap='gray')
    plt.title('impronta 2')
    plt.show()