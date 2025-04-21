import numpy as np 
import matplotlib.pyplot as plt
import scipy.ndimage as ndi 
import skimage.io as io 


def bit_get(x, index):
    if x.dtype not in [np.uint8, np.uint16, np.uint32, np.uint64]:
        raise TypeError('Errore nel tipo di codifica dell immagine')
    else:
        return ((1 << index) & x) != 0

def MSE(x,y):
    return np.mean((x-y)**2)

def test(x):
    M,N = 512,512
    x = x[:(M-1), :(N-1)]
    B2 = np.int8(bit_get(x, 1))
    f1 = np.sum(np.abs(B2[1:,:]-B2[:-1,:]))/((M-1)*N)
    f2 = np.sum(np.abs(B2[:,1:], B2[:,:-1]))/(M*(N-1))

    y = ndi.gaussian_filter(x, 0.5)
    X = np.fft.fft2(x)
    Y = np.fft.fft2(y)  

    f3 = 10*np.log10(MSE(np.abs(X), np.abs(Y)))
    f4 = 10*np.log10(MSE(np.angle(X), np.angle(Y)))

    f = 0.7*f1 + 1.5*f2 + 0.01*f3 + 0.001*f4
    
    if f > 1.5:
        return 'camera 1'
    else:
        return 'camera 2'

if __name__ == '__main__':
    path = 'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\esercitazioni\\simulazione_11_04_25\\esercitazione_250411\\immagini\\'

    ims = [path+'1.png',path+'2.png',path+'3.png',path+'4.png']

    x = []

    for im in ims:
        x.append(np.uint8(io.imread(im)))
    
    for image in x:
        
        plt.figure()
        plt.imshow(image, clim=[0,255], cmap='gray')
        plt.title(test(image))
    
    plt.show()