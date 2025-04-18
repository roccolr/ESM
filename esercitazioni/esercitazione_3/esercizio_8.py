import numpy as np 
import skimage.io as io 
import skimage.util as ut
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import sys 
sys.path.append('C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio')
from my_modules.my_lib import rotate
path = 'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\Immagini\\'

def generate_noise(x, dev):
    r = dev*np.random.randn(x.shape[0], x.shape[1])
    return r

def generate_masks(n:int,s:int):
    """
    Ritorna 4 maschere booleane di dimensione n
    """

    mask_3 = np.triu(np.ones((n,n)), -s) - np.triu(np.ones((n,n)), s+1)
    mask_1 = mask_3[:,::-1]
    mask_2 = np.zeros((n,n))
    mask_2[n//2-s : n//2+s+1]=1
    mask_4 = mask_2.T
    masks = np.stack((mask_1, mask_2, mask_3, mask_4), 0)
    return masks>0

def custom_fun(x, masks):
    K, M, N = masks.shape
    block = x.reshape([M,N]) # da vettore a matrice

    varianze = np.zeros(K)
    for k in range(K):
        varianze[k] = np.var(block[masks[k]])
    
    idx = np.argmin(varianze)

    # filtraggio
    value = np.mean(block[masks[idx]])
    return value


if __name__ == '__main__':

    # lettura immagini
    im = path + 'zebre.y'
    x = np.float32(np.fromfile(im, dtype=np.uint8))
    x = np.reshape(x, (321,481))

    #generazione rumore
    noise = generate_noise(x,25)
    noisy_x = x+noise

    #filtraggio
    masks = generate_masks(9, 1)
    y = ndi.generic_filter(noisy_x, custom_fun, (9,9), mode='reflect', extra_keywords={"masks":masks})


    # stampa section
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.imshow(x, clim=[0,255], cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(noisy_x, clim=[0,255], cmap='gray')
    plt.figure(2)
    plt.imshow(y, clim=None, cmap='gray')
    plt.colorbar()
    plt.title('Filtered')
    plt.show()