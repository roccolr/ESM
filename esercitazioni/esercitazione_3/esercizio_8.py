import numpy as np 
import skimage.io as io 
import skimage.util as ut
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import sys 
sys.path.append('C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio')
from my_modules.my_lib import rotate
path = 'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\Immagini\\'

m1 = np.array([[0,0,0,0,0,0,0,1,1],
                [0,0,0,0,0,0,1,1,1],
                [0,0,0,0,0,1,1,1,0],
                [0,0,0,0,1,1,1,0,0],
                [0,0,0,1,1,1,0,0,0],
                [0,0,1,1,1,0,0,0,0],
                [0,1,1,1,0,0,0,0,0],
                [1,1,1,0,0,0,0,0,0],
                [1,1,0,0,0,0,0,0,0],
                ],dtype=np.float32)

m2 = np.array([[0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [1,1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1,1],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                ],dtype=np.float32)

m3 = rotate(m1, np.pi/2)

m4 = rotate(m2, np.pi/2)

masks = [m1,m2,m3,m4]

def generate_noise(x):
    r = 25*np.random.randn(x.shape[0], x.shape[1])
    return r

def my_mean(x):
    sum = 0
    denom = -1
    for elem in x:
        sum+=elem
        if(elem != 0):
            denom+=1
    return float(sum/denom)

def my_var(x):
    x = x.reshape(81)
    sum = 0
    denom = 0
    avg = my_mean(x)
    for elem in x:
        if elem != 0:
            sum += (elem - avg)**2
            denom += 1
    return np.float32(sum/denom)



def local_var(x):
    if x.size != 81:
        print('valore incompatibile')
        return 0  # oppure np.mean(x), oppure qualche valore di default
    x = x.reshape((9, 9))
    data = []
    for m in masks:
        var = my_var(x * m)
        data.append((np.var(x), m))
    mask = min(data, key=lambda t: t[0])[1]
    return my_mean((x*mask).reshape(81))
    # data = []
    # x = x.reshape((9, 9))
    # try:
    #     for m in masks:
    #         data.append((np.var(x*m),m))
    # except ValueError as e:
    #     print(e)
    # mask = min(data, key= lambda t: t[0])[1]
    # print(np.mean(x*mask)) 

if __name__ == '__main__':
    im = path + 'zebre.y'
    x = np.float32(np.fromfile(im, dtype=np.uint8))
    x = np.reshape(x, (321,481))
    noise = generate_noise(x)
    noisy_x = x+noise

    y = ndi.generic_filter(noisy_x, local_var, (9,9), mode='constant')


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