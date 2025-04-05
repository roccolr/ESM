# porre a zero i bit usando il pit plane slicing 

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi 
import skimage.io as io 
import sys
sys.path.insert(1,'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio')
import immagini.Immagini.bitop as bit 
import my_modules.histogramop as hope

path = 'C:\\Users\\rocco\\Documents\\università\ESM\\laboratorio\\immagini\\Immagini\\lena.y'
marchio = 'C:\\Users\\rocco\\Documents\\università\ESM\\laboratorio\\immagini\\Immagini\\marchio.y'

lena = np.fromfile(path, np.uint8)
lena = np.reshape(lena, (512,512))
lena = lena[0:350, 0:350]
m = np.fromfile(marchio, np.uint8)
m = np.reshape(m, (350,350))
m = hope.fshs(m, 256)
mod_lena = bit.bitset(lena, 2, m)

plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(lena, clim=[0,255], cmap='gray')
plt.subplot(1,2,2)
plt.imshow(mod_lena, clim=[0,255], cmap='gray')
plt.show()
