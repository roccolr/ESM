import numpy as np 
import matplotlib.pyplot as plt
import skimage.io as io 
import scipy.ndimage as ndi

path = 'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\Immagini\\'

def genera_filtri():
    filters = []
    filters.append(np.ones((5,5), dtype=np.float32)/(5**2))
    filters.append(np.ones((10,10), dtype=np.float32)/(10**2))
    filters.append(np.ones((16,16), dtype=np.float32)/(15**2))
    return filters


if __name__ == '__main__':
    filters = genera_filtri()
    lena = path + 'lena.jpg'
    x = np.float32(io.imread(lena))
    X = np.fft.fft2(x)


    P1 = int(102.4*filters[0].shape[0])
    Q1 = int(102.4*filters[0].shape[1])
    H1 = np.fft.fft2(filters[0], (P1,Q1))
    P2 = int(51.2*filters[1].shape[0])
    Q2 = int(51.2*filters[1].shape[1])
    H2 = np.fft.fft2(filters[1], (P2,Q2))
    P3 = 32*filters[2].shape[0]
    Q3 = 32*filters[2].shape[1]
    H3 = np.fft.fft2(filters[2], (P3,Q3))

    plt.figure()
    plt.imshow(np.log(1+np.abs(np.fft.fftshift(H1))), clim=None, cmap='gray')
    plt.title('filtro 5x5')
    plt.figure()
    plt.imshow(np.log(1+np.abs(np.fft.fftshift(H2))), clim=None, cmap='gray')
    plt.title('filtro 10x10')
    plt.figure()
    plt.imshow(np.log(1+np.abs(np.fft.fftshift(H3))), clim=None, cmap='gray')
    plt.title('filtro 15x15')
    plt.figure()
    plt.imshow(np.log(np.abs(np.fft.fftshift(X))+1), clim=None, cmap='gray')
    plt.title('trasformata di Lena')

    # verifica del comportamento passa basso del filtro

    Y1 = X*H1
    Y2 = X*H2 
    Y3 = X*H3

    y1 = np.real(np.fft.ifft2(Y1))
    y2 = np.real(np.fft.ifft2(Y2))
    y3 = np.real(np.fft.ifft2(Y3))

    plt.figure()
    plt.imshow(y1, clim=None, cmap='gray')
    plt.title('lena_filtro_1')
    plt.figure()
    plt.imshow(y2, clim=None, cmap='gray')
    plt.title('lena_filtro_2')
    plt.figure()
    plt.imshow(y3, clim=None, cmap='gray')
    plt.title('lena_filtro_3')

    plt.show()