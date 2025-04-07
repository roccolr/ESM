import numpy as np 
import skimage.io as io 
import skimage.util as ut
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import sys 
sys.path.append('C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio')
from my_modules.my_lib import rotate
path = 'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\Immagini\\'


def filtro_guidate(x,g,B):
    # normalizzazione
    e = 2**(-60)
    x_max = np.max(x)
    g_max = np.max(g)
    x = x/x_max
    g = g/g_max

    m_x = ndi.generic_filter(x, np.mean, (B,B))
    m_g = ndi.generic_filter(g, np.mean, (B,B))

    v_x = ndi.generic_filter(x, np.var, (B,B))
    v_g = ndi.generic_filter(g, np.var, (B,B))

    c_gx = ndi.generic_filter(x*g, np.sum, (B,B))/B**2

    a = (c_gx - m_x*m_g)/(v_g+e)
    b = m_x - a*m_g

    mu_a = ndi.generic_filter(a, np.mean, (B,B))
    mu_b = ndi.generic_filter(b, np.mean, (B,B))

    return mu_a*g + mu_b

if __name__ == '__main__':
    imm1 = path+'mask.png'
    imm2 = path+'guida.png'
    mappa = np.float32(io.imread(imm1))
    guida = np.float32(io.imread(imm2))
    B = 10
    nuova_mappa = filtro_guidate(mappa, guida, B)
    # stampa section
    plt.figure(1)
    plt.imshow(mappa, clim=[0,255], cmap='gray')
    plt.title('maschera')
    plt.colorbar()
    plt.figure(2)
    plt.imshow(guida, clim=None, cmap='gray')
    plt.title('guida')
    plt.colorbar()
    plt.figure(3)
    plt.imshow(nuova_mappa, clim=None, cmap='gray')
    plt.title('nuova_mappa')
    plt.figure(4)
    plt.imshow(mappa*guida, clim=None, cmap='gray')
    plt.title('x*g')
    plt.figure(5)
    plt.imshow(nuova_mappa*guida, clim=None, cmap='gray')
    plt.title('y*g')
    plt.show()