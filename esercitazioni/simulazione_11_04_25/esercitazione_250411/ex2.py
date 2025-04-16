import numpy as np 
import matplotlib.pyplot as plt 
import scipy.ndimage as ndi
import skimage.io as io 

path = "C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\esercitazioni\\simulazione_11_04_25\\esercitazione_250411\\"

def local_var(x, size:int):
    return ndi.generic_filter(x, np.var, (size,size))

def my_filter(x, soglia):
    k = int((x.shape[0])**0.5)
    x = np.reshape(x, (k,k))
    var = np.var(x)
    if var < soglia:
        return np.mean(x)
    else:
        x = x[1:k-1,1:k-1]
        return np.mean(x)
    return 0

def add_noise(x, sigma):
    n = sigma*np.random.randn(x.shape[0], x.shape[1])
    return x + n

def adapt_filter(x, k):
    var_l = local_var(x,k)
    var_lf = var_l.flatten()
    sorted_var = np.sort(var_lf)
    T = sorted_var[int(0.7*len(sorted_var))]
    return ndi.generic_filter(x, my_filter, size=(k,k), extra_arguments=(T,))

def MSE(x,y):
    return np.mean((x-y)**2)

def PSNR(x,y):
    p = np.max(x)
    return 10*np.log10(p**2/MSE(x,y))


if __name__ == '__main__':
    im = path + 'cigno.jpg'

    x = np.float32(io.imread(im))
    y = add_noise(x, 25)
    k = 7

    k_list = [3,5,7,9]
    PSNR_list = []
    AVG_list = []

    plt.close('all')
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.imshow(x, clim=[0,255], cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(y, clim=[0,255], cmap='gray')
    for item in k_list:
        z = adapt_filter(y, item)
        m = ndi.uniform_filter(y, item, mode='reflect')
        PSNR_list.append(PSNR(x, z))
        AVG_list.append(PSNR(x,m))
        if(item == 5):
            plt.figure()
            plt.imshow(z, clim=None, cmap='gray')
            plt.title(f'k = {item}')
            plt.figure()
            plt.imshow(m, clim=None, cmap='gray')
    
    plt.figure()
    plt.plot(k_list, PSNR_list)
    plt.grid()
    plt.title('PSNR')
    plt.plot(k_list, AVG_list)
    plt.grid()
    plt.title('PSNR ignorante')
    plt.show()

