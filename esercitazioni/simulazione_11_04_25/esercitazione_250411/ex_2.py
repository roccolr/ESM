import numpy as np 
import matplotlib.pyplot as plt
import skimage.io as io 
import scipy.ndimage as ndi 

def add_gaussian_noise(x, sigma):
    noise = sigma*np.random.randn(x.shape[0], x.shape[1])
    return noise + x

def custom_filter(x, th):
    if np.var(x) < th:
        return np.mean(x)
    else:
        k = np.int32((len(x))**0.5)
        x = np.reshape(x, (k,k))
        return np.mean(x[1:-1, 1:-1])

def adapt_filter(x,k):
    loc_var = ndi.generic_filter(x, np.var, (k,k))
    var_vector = np.reshape(loc_var, (loc_var.shape[0]*loc_var.shape[1],1))
    var_vector = np.sort(var_vector)
    th = np.float32(var_vector[np.int32(len(var_vector)*0.7)])
    y = ndi.generic_filter(x, custom_filter, (k,k), extra_keywords={'th':th})
    return y

def PSNR(x,y):
    max_x = np.max(x)
    return 10*np.log10((max_x**2)/(np.mean((x-y)**2)))

if __name__ == '__main__':
    path = 'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\esercitazioni\\simulazione_11_04_25\\esercitazione_250411\\immagini\\'

    im = path + 'cigno.jpg'
    x = np.float32(io.imread(im))
    noisy = add_gaussian_noise(x,25)
    y = adapt_filter(noisy, 5)

    values = [3,5,7,9]
    for value in values:
        print(f'[PSNR k={value}: {PSNR(x,adapt_filter(x,value))}]')

        
    plt.figure(1)
    plt.imshow(x, clim=[0,255], cmap='gray')
    plt.colorbar()
    plt.title('input')
    plt.figure(2)
    plt.imshow(noisy, clim=[0,255], cmap='gray')
    plt.colorbar()
    plt.title('noisy input')
    plt.figure(3)
    plt.imshow(y, clim=None, cmap='gray')
    plt.colorbar()
    plt.title('output')
    plt.show()