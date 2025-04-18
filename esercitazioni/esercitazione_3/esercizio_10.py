import numpy as np 
import skimage.io as io 
import skimage.util as ut
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import sys 
sys.path.append('C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio')
from my_modules.my_lib import rotate
path = 'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\Immagini\\'

V = np.array([[0,-1,0],[0,1,0],[0,0,0]],dtype=np.float32) # derivata direzione verticale
H = np.array([[0,0,0],[-1,1,0],[0,0,0]],dtype=np.float32) # derivata direzione orizzontale
D1 = np.array([[-1,0,0],[0,1,0],[0,0,0]],dtype=np.float32) # derivata direzione obliqua 1
D2 = np.array([[0,0,0],[0,1,0],[-1,0,0]],dtype=np.float32) # derovata direzione obliqua 2


def extract_keypoint(x):
    d_v = ndi.correlate(x,V,mode='reflect')
    d_h = ndi.correlate(x,H,mode='reflect')
    d_o1 = ndi.correlate(x,D1, mode='reflect')
    d_o2 = ndi.correlate(x,D2, mode='reflect')

    m_v = ndi.generic_filter(d_v**2, np.mean, (5,5))
    m_h = ndi.generic_filter(d_h**2, np.mean, (5,5))
    m_o1 = ndi.generic_filter(d_o1**2, np.mean, (5,5))
    m_o2 = ndi.generic_filter(d_o2**2, np.mean, (5,5))

    stacked = np.stack((m_v,m_h,m_o1,m_o2),axis=-1)
    Q_min = np.min(stacked,-1)
    MQ_min = ndi.generic_filter(Q_min, np.max, (3,3))

    SP =((Q_min>500) & (Q_min == MQ_min))
    return SP

if __name__=='__main__':
    im = path + 'tetto.png'
    x = np.float32(io.imread(im))

    map = extract_keypoint(x)


    # stampa section
    plt.close('all')
    plt.figure(1)
    plt.imshow(x, clim=[0,255], cmap='gray')
    plt.colorbar()
    plt.title('originale')
    plt.figure(2)
    plt.imshow(map, clim=[0,1], cmap='gray')
    plt.colorbar()
    plt.title('key_points')
    plt.show()