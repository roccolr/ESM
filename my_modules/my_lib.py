import numpy as np 
import matplotlib.pyplot as plt
import scipy.ndimage as ndi 
import skimage.io as io 
import random

def readjpg(nomefile:str):
    """
    Shows an image from a jpeg file
    """
    x = io.imread(nomefile)
    plt.figure() # apre una nuova figura
    plt.imshow(x, clim=[0,255], cmap = 'gray')

def readraw(nomefile:str, nrighe, ncolonne, tipo):
    """
    Shows an image from a raw file
    """
    x = np.fromfile(nomefile, tipo)
    x = np.reshape(x, (nrighe, ncolonne))
    plt.figure() # apre una nuova figura
    plt.imshow(x, clim=[0,255], cmap = 'gray')
    
def clip(x:np.array, limit:int) -> np.array:
    """
    Returns a clipped copy of the input array
    """
    y = []
    for elem in x:
        if abs(elem) > limit:
            if elem>0:
                y.append(limit)
            else:
                y.append(-limit)
        else:
            y.append(elem)
    
    return np.array(y)

def clip(x:np.array) -> np.array:
    """
    Given a 16x16 matrix, returns a matrix containing the 4 greatest values
    """
    N = x.shape[0]
    M = x.shape[1]
    y = np.reshape(x,M*N)
    y = np.sort(y)
    greatest_values = y[-4:]
    # print(greatest_values)
    trashold = np.min((greatest_values))
    print(f'Trashold set:\t{trashold}')

    for i in range(0, N):
        for j in range(0, M):
            if x[i,j] < trashold:
                x[i,j] = 0
    
    return x

def create_random_matrix(dim:tuple) -> np.array:
    N = dim[0]
    M = dim[1]

    elements = []
    for i in range(0, N):
        sublist = []
        for j in range(0, M):
            sublist.append(int(random.random()*100))
        elements.append(sublist)
    
    return np.array(elements)

def local_enhance(x:np.array, k1:float, k2:float, k3:float, gain:float) -> tuple:

    """
    Returns the enhanced image and the mask

    Parameters:
        k1 (float): Superior limit for avg
        k2 (float): Superior limit for std
        k3 (float): Inferior limit for std
        gain (float): gain to selected pixels

    Returns:
        tuple: image after manipulation, mask
    """

    g_mean = np.mean(x)
    g_std = np.std(x)

    l_mean = ndi.generic_filter(x, np.mean, (3,3))
    l_dev = ndi.generic_filter(x, np.std, (3,3))

    m = (l_mean <= k1 * g_mean) & (l_dev <= k2 * g_std) & (l_dev >= k3*g_std)
    y = gain*m*x + (1-m)*x
    return y,m
