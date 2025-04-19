# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 11:46:55 2025

@author: Davide
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import skimage.io as io

def vediJpeg(nomefile):
    x = io.imread(nomefile)
    plt.figure()
    plt.imshow(x, clim=[0,255], cmap='gray')
    
def vediRAW(nomefile, nRighe, nColonne, tipo):
    x = np.fromfile(nomefile, tipo)
    x = np.reshape(x, (nRighe, nColonne))
    plt.figure()
    plt.imshow(x, clim=[0,255], cmap='gray')
    
def fshs(x, K=256):
    xmin = np.min(x)
    xmax = np.max(x)
    y = (K-1) * (x-xmin) / (xmax-xmin)
    return y


def rgb2cmy(x):
    #C = 1 - x[:,:,0]
    #M = 1 - x[:,:,1]
    #Y = 1 - x[:,:,2]
    #y = np.stack((C,M,Y),2)
    y = 1-x
    return y

def cmy2rgb(x):
    y = 1-x
    return y