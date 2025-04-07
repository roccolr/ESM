import numpy as np 
import skimage.io as io 
import skimage.util as ut
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import sys 
sys.path.append('C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio')
from my_modules.my_lib import rotate
path = 'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\Immagini\\'

m1 = np.array([[0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0],
              [1,1,1,1,1,1,1],
              [0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0]
              ])
m2 = np.array([[0,0,0,1,0,0,0],
              [0,0,0,1,0,0,0],
              [0,0,0,1,0,0,0],
              [0,0,0,1,0,0,0],
              [0,0,0,1,0,0,0],
              [0,0,0,1,0,0,0],
              [0,0,0,1,0,0,0]
              ])

m3 = np.array([[1,0,0,0,0,0,0],
              [0,1,0,0,0,0,0],
              [0,0,1,0,0,0,0],
              [0,0,0,1,0,0,0],
              [0,0,0,0,1,0,0],
              [0,0,0,0,0,1,0],
              [0,0,0,0,0,0,1]
              ])

m4 = np.array([[0,0,0,0,0,0,1],
              [0,0,0,0,0,1,0],
              [0,0,0,0,1,0,0],
              [0,0,0,1,0,0,0],
              [0,0,1,0,0,0,0],
              [0,1,0,0,0,0,0],
              [1,0,0,0,0,0,0]
              ])

def my_filter(x):
    x = x.reshape((7,7))
    data1 = np.mean(x*m1) - np.mean(x*(1-m1))
    data2 = np.mean(x*m2) - np.mean(x*(1-m2))
    data3 = np.mean(x*m3) - np.mean(x*(1-m3))
    data4 = np.mean(x*m4) - np.mean(x*(1-m4))

    stacked = np.stack((data1,data2,data3,data4), axis=-1)
    return np.min(stacked, axis=-1)

def segmenta(x):
    # x = banda verde dell'immagine
    filtered_x = ndi.generic_filter(x, my_filter, (7,7))
    map = filtered_x > -5
    return map

if __name__=='__main__':
    im = path + 'retina.tif'
    
    x = np.float32(io.imread(im, plugin='pil'))
    # x = np.reshape(x, (565,584,3))

    green_x = x[:,:,1]

    map = segmenta(green_x)

    # stampa section
    plt.close('all')
    plt.figure(1)
    plt.subplot(1,4,1)
    plt.imshow(m1,clim=[0,1], cmap='gray')
    plt.subplot(1,4,2)
    plt.imshow(m2,clim=[0,1], cmap='gray')
    plt.subplot(1,4,3)
    plt.imshow(m3,clim=[0,1], cmap='gray')
    plt.subplot(1,4,4)
    plt.imshow(m4,clim=[0,1], cmap='gray')
    plt.figure(2)
    plt.imshow(green_x,clim=None, cmap='gray')
    plt.figure(3)
    plt.imshow(map,clim=[0,1], cmap='gray')
    plt.show()