import numpy as np 
import skimage.io as io 
import skimage.util as ut
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import sys
sys.path.append('C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio')
from my_modules.seg_utils import zero_crossing
path = 'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\Immagini\\'

def enhanc(x,mask,k):
    """
    Returns the enhanced image

    Parameters:
        x: Input Image
        mask: Mask that localizes the imperfections
        k: The process will be iterated k times

    Returns:
        np.Array: output enhanced image  
    """
    a = 0.073
    b = 0.177
    h = np.array([[a,b,a],[b,0,b], [a,b,a]])
    y = np.copy(x)
    for i in range(0,k):
        z = ndi.correlate(y,h,mode='reflect')
        y = (1-mask)*z + mask*y
    
    return y


if __name__ == '__main__':
    im = path+'bebe.jpg'
    mask = path+'mask.bmp'

    x = np.float32(io.imread(im))
    m = np.float32(io.imread(mask))

    y= enhanc(x, m, 100)
    z = enhanc()

    plt.close('all')
    plt.figure(1)
    plt.imshow(x, clim=[0,255], cmap='gray')
    plt.colorbar()
    plt.title('input')
    plt.figure(2)
    plt.imshow(m, clim= [0,1], cmap='gray')
    plt.colorbar()
    plt.title('mask')
    plt.figure(3)
    plt.imshow(y, clim= [0,255], cmap='gray')
    plt.colorbar()
    plt.title('enhanced')
    plt.show()