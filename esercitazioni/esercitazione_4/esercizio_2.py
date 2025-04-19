import numpy as np 
import skimage.io as io 
import skimage.util as ut
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
path = 'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\Immagini\\'

if __name__ == '__main__':
    im = path + 'volto.tif'
    x = np.float32(io.imread(im, plugin='pil'))

    X = np.fft.fft2(x)
    A_X = np.log(1+np.abs(np.fft.fftshift(X)))      # ampiezza
    F_X = np.log(1+np.angle(np.fft.fftshift(X)))    # fase

    X1 = np.abs(X)                  # prendiamo la sola componente relativa al modulo
    X2 = np.exp(1j*np.angle(X))     # prendiamo la sola componente relativa alla fase

    y1 = np.real(np.fft.ifft2(X1)) # solo parte reale per eventuali errori di calcolo
    y2 = np.real(np.fft.ifft2(X2))    

    plt.close('all')
    plt.figure(1)
    plt.imshow(x, clim=[0,255], cmap='gray')
    plt.title('originale')
    plt.colorbar()
    plt.figure(2)
    plt.imshow(A_X, clim=None, cmap='gray', extent=(-0.5,+0.5,+0.5,-0.5))
    plt.title('spettro di ampiezza')
    plt.colorbar()
    plt.figure(3)
    plt.imshow(F_X, clim=[-np.pi,np.pi], cmap='gray', extent=(-0.5,+0.5,+0.5,-0.5))
    plt.title('spettro di fase')
    plt.colorbar()
    plt.figure(4)
    plt.subplot(1,2,1)
    plt.imshow((y1-np.min(y1))**0.1, clim=None, cmap='gray')
    plt.title('ricostruzione modulo')
    plt.subplot(1,2,2)
    plt.imshow(y2, clim=None, cmap='gray')
    plt.title('ricostruzione fase')
    plt.figure(5)
    plt.imshow(np.real(np.fft.ifft2(X)), clim=[0,255], cmap='gray')
    plt.title('ricostruzione completa')
    plt.show()