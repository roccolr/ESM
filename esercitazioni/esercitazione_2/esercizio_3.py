# porre a zero i bit usando il pit plane slicing 

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi 
import skimage.io as io 
import sys
sys.path.insert(1,'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio')
import immagini.Immagini.bitop as bit 

path = 'C:\\Users\\rocco\\Documents\\università\ESM\\laboratorio\\immagini\\Immagini\\frattale.jpg'

x = np.uint8(io.imread(path))
y = x.copy()
for i in range(5):
    y = bit.bitset(y, i, 0)

plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(x, cmap='gray', clim=[0,255])
plt.subplot(1,2,2)
plt.imshow(y, cmap='gray', clim=[0,255])
plt.show()

# posso rimuovere fino ai 4 livelli meno significativi per avere risultati accettabili di compressione