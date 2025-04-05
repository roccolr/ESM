import numpy as np 
import matplotlib.pyplot as plt
import scipy.ndimage as ndi 
import skimage.io as io 

def medie(x:np.array, k:int)->np.array:
    """
    Restituisce l'immagine delle medie locali su finestre KxK
    """
    y = ndi.generic_filter(x, np.mean, (k,k))
    return y


if __name__ == '__main__':
    path = 'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\immagini\\Immagini\\azzurro.jpg'
    x = io.imread(path)

    R0 = x[:,:,0]
    G0 = x[:,:,1]
    B0 = x[:,:,2]

    R1 = medie(R0, 50)
    G1 = medie(G0, 50)
    B1 = medie(B0, 50)

    y = np.stack((R1,G1,B1), -1)

    plt.figure(1)
    plt.subplot(1,2,1)
    plt.imshow(x)
    plt.title('original')
    plt.subplot(1,2,2)
    plt.imshow(y)
    plt.title('elaborated')
    plt.show()

