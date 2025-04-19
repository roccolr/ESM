import numpy as np 
import matplotlib.pyplot as plt
import skimage.io as io 
import scipy.ndimage as ndi 

def detect(x):
    h_hori = np.array([[0,0,0],[-1,2,-1],[0,0,0]], dtype=np.float32)
    d2_hori = ndi.correlate(x,h_hori, mode='reflect')
    M,N = x.shape
    v_hori = np.sum(np.abs(d2_hori), axis=0)
    d_hori = v_hori[1:] - v_hori[:-1]
    D_hori = np.abs(np.fft.fftshift(np.fft.fft(d_hori, N-2)))
    f_hori = np.fft.fftshift(np.fft.fftfreq(N-2))
    ni_hori = f_hori[(D_hori == np.max(D_hori)) & (f_hori>0)]
    R_hori = 1/ni_hori

    d2_vert = ndi.correlate(x,h_hori.T, mode='reflect')
    v_vert = np.sum(np.abs(d2_vert), axis=1)
    d_vert = v_vert[1:] - v_vert[:-1]
    D_vert = np.abs(np.fft.fftshift(np.fft.fft(d_vert, M-2)))
    f_vert = np.fft.fftshift(np.fft.fftfreq(M-2))
    ni_vert = f_vert[(D_vert==np.max(D_vert)) & (f_vert>0)]
    R_vert = 1/ni_vert

    return D_hori, D_vert, R_hori, R_vert



if __name__ == '__main__':
    path = 'C:\\Users\\rocco\\Documents\\università\\ESM\\laboratorio\\Immagini\\'
    x = np.fromfile(path+'zoom.y', np.float32)
    x = x.reshape((128,128))
    D_hori, D_vert, R_hori, R_vert = detect(x)
    print(f'D_hori = {D_hori}, D_vert = {D_vert}, R_hori= {R_hori}, R_vert = {R_vert}')

    f_hori = np.fft.fftshift(np.fft.fftfreq(len(D_hori)))
    f_vert = np.fft.fftshift(np.fft.fftfreq(len(D_vert)))
    plt.figure() 
    plt.subplot(2,1,1)
    plt.plot(f_hori, D_hori,'-or')
    plt.title(f'DFT derivata pseudo-varianza orizontale, fattori di scala: {R_hori}')
    plt.subplot(2,1,2)
    plt.plot(f_vert, D_vert,'-or')
    plt.title(f'DFT derivata pseudo-varianza verticale, fattori di scala: {R_vert}')

    plt.show()