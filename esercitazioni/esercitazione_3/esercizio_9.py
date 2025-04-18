import numpy as np 
import skimage.io as io 
import skimage.util as ut
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import sys 
sys.path.append('C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio')
from my_modules.my_lib import rotate
path = 'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\Immagini\\'


def filtro_guidato(x,g,B):
    med_x = ndi.uniform_filter(x, B, mode='reflect')
    med_g = ndi.uniform_filter(g, B, mode='reflect')
    var_g = ndi.generic_filter(g, np.var, (B,B), mode='reflect')
    corr_gx = ndi.uniform_filter(g*x, (B,B))
    eps = 2**(-60)

    a = (corr_gx-med_x*med_g)/(var_g+eps)
    b = med_x - a*med_g

    med_a = ndi.uniform_filter(a, (B,B))
    med_b = ndi.uniform_filter(b, (B,B))
    return med_a*g + med_b


if __name__ == '__main__':
    imm1 = path+'mask.png'
    imm2 = path+'guida.png'
    mappa = np.float32(io.imread(imm1))/255
    guida = np.float32(io.imread(imm2))/255
    B = 10
    nuova_mappa = filtro_guidato(mappa, guida, B)
    # stampa section
    plt.figure(1)
    plt.imshow(mappa, clim=[0,1], cmap='gray')
    plt.title('maschera binaria')
    plt.colorbar()
    plt.figure(2)
    plt.imshow(guida, clim=[0,1], cmap='gray')
    plt.title('immagine guida')
    plt.colorbar()
    plt.figure(3)
    plt.imshow(nuova_mappa, clim=None, cmap='gray')
    plt.title('maschera elaborata')
    plt.figure(4)
    plt.imshow(mappa*guida, clim=None, cmap='gray')
    plt.title('x*g')
    plt.figure(5)
    plt.imshow(nuova_mappa*guida, clim=None, cmap='gray')
    plt.title('y*g')
    plt.show()