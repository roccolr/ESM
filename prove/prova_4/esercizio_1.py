# -*- coding: utf-8 -*-
"""
@author: %Bombombini Goosini
"""
import numpy as np
import matplotlib.pyplot as plt 
import skimage.io as io 


def bitset(x, index, v):
    """
    Return an unsigned integer array modifying a bit-plane of a given array.

    Parameters
    ----------
    x : unsigned integer array 
        Modify the bit-plane from this array.
    index : int
        Posizion of bit-plane to modify.
        It must be less than number of bits in the integer class of x.
    v : 0 or 1 or boolean arry
        Value or values used to modify the bit-plane.
        The bits where v is False are set to 0 (off), and
        The bits where v is True are set to 1 (on).

    Returns
    -------
    unsigned integer array of the same shape of x
        Array with the modified bit-plane.

    Example
    -------
    a = bitset(x,0,1)
    
    """
    
    import numpy
    if x.dtype not in [numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64]:
        raise ValueError("Only uint8, uint16, uint32, and uint64 are supported as dtype!")
    if numpy.isscalar(v):
        v = numpy.asarray([v!=0,], dtype=x.dtype)
    else:
        v = numpy.asarray(v!=0, dtype=x.dtype)
    vp = v << index
    vn = ~((1-v) << index)
    
    return (x & vn) | vp 

def MSE(x,y):
    return np.mean((x-y)**2)

if __name__ == '__main__':
    path = 'C:/Users/Flexo Rodriguez/Desktop/ESM/ESM/prove/immagini/'
    
    x = np.reshape(np.uint8(np.fromfile(path+'upupa.y', dtype=np.uint8)), (256,512))
    f = np.reshape(np.uint8(np.fromfile(path+'firma.y', dtype=np.uint8)), (256,512))
    
    y = bitset(x, 1, f)
    
    Q_list = [80, 90, 100]
    MSE_list = []
    i = 0
    for Q in Q_list:
        io.imsave(f'C:/Users/Flexo Rodriguez/Desktop/ESM/ESM/prove/prova_4/temp/im_{i}.jpeg', y, quality=Q)
        z = np.float32(io.imread(f'C:/Users/Flexo Rodriguez/Desktop/ESM/ESM/prove/prova_4/temp/im_{i}.jpeg'))
        MSE_list.append(MSE(x,z))
        i+=1
    
    
    Y = np.fft.fftshift(np.fft.fft2(x))
    
    # filtraggio 
    M,N = y.shape
    m = np.fft.fftshift(np.fft.fftfreq(M))
    n = np.fft.fftshift(np.fft.fftfreq(N))
    l,k = np.meshgrid(n,m)
    
    B_list = [0.2,0.3,0.4]
    MSE_list_2 = []
    for B in B_list:
        H = np.sqrt(k**2+l**2) < B    
        Z = Y*H
        zt = np.real(np.fft.ifft2(np.fft.ifftshift(Z)))
        MSE_list_2.append(MSE(x,zt))

    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(x, clim=[0,255], cmap = 'gray')
    plt.title('input')
    plt.subplot(1,2,2)
    plt.imshow(f,clim=[0,1], cmap='gray')
    plt.title('firma')   
    plt.figure()
    plt.imshow(y, clim=[0,255], cmap='gray')
    plt.title('input con firma')    
    plt.figure()
    plt.plot(Q_list, MSE_list)
    plt.xlabel('Q')
    plt.ylabel('MSE')
    plt.title('MSE con std jpg')
    
    plt.figure()
    plt.plot(B_list, MSE_list_2)
    plt.xlabel('Q')
    plt.ylabel('MSE')
    plt.title('MSE con trasformata')
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    