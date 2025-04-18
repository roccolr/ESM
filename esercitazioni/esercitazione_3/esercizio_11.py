import numpy as np 
import skimage.io as io 
import skimage.util as ut
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import sys 
sys.path.append('C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio')
from my_modules.my_lib import rotate
path = 'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\Immagini\\'

def genera_maschera(n):
    m1 = np.zeros((n,n))
    m1[n//2] = 1
    m2 = m1.T
    m3 = np.triu(np.ones((n,n)), 0) - np.triu(np.ones((n,n)), 1)
    m4 = m3[:,::-1]
    
    masks = np.stack((m1,m2,m3,m4), 0)
    return masks>0

def my_filter(x, masks):
    x = x.reshape((7,7))
    diff_0 = np.mean(x[masks[0]]) - np.mean(x[1-masks[0]])
    diff_1 = np.mean(x[masks[1]]) - np.mean(x[1-masks[1]])
    diff_2 = np.mean(x[masks[2]]) - np.mean(x[1-masks[2]])
    diff_3 = np.mean(x[masks[3]]) - np.mean(x[1-masks[3]])
    stacked_diff = np.stack((diff_0, diff_1, diff_2, diff_3), -1)
    min = np.min(stacked_diff, -1)
    return min

def segmenta(x, masks):
    filtered_x = ndi.generic_filter(x, my_filter, (7,7), extra_keywords={'masks':masks})
    map = filtered_x > -5
    return map

def elimina_cerchio(x):
    M,N = x.shape
    j,i = np.meshgrid(np.arange(N)-N/2, np.arange(M)-M/2)
    m = (j**2 + i**2) < 0.2*N*M
    return x*m

if __name__=='__main__':
    im = path + 'retina.tif'
    
    x = np.float32(io.imread(im, plugin='pil'))
    green_x = x[:,:,1]
    masks = genera_maschera(7)
    map = segmenta(green_x, masks)
    clear_map = elimina_cerchio(map)
    # stampa section
    plt.close('all')
    plt.figure(1)
    plt.subplot(1,4,1)
    plt.imshow(masks[0],clim=[0,1], cmap='gray')
    plt.subplot(1,4,2)
    plt.imshow(masks[1],clim=[0,1], cmap='gray')
    plt.subplot(1,4,3)
    plt.imshow(masks[2],clim=[0,1], cmap='gray')
    plt.subplot(1,4,4)
    plt.imshow(masks[3],clim=[0,1], cmap='gray')
    plt.figure(2)
    plt.imshow(green_x,clim=None, cmap='gray')
    plt.figure(3)
    plt.imshow(map,clim=[0,1], cmap='gray')
    plt.figure(4)
    plt.imshow(clear_map,clim=[0,1], cmap='gray')
    plt.show()