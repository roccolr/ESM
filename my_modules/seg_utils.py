# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 10:07:33 2020

@author: Davide
"""

import numpy as _np
from numpy import abs as _abs

def zero_crossing(b, thresh=None):
    """
    Restituisce la mappa dei passaggi per zero.

    Parameters
    ----------
    b : matrice 
    thresh : soglia opsionale

    Returns
    -------
    m : mappa dei zero crossing

    """
    if thresh is None:
        thresh = 0.25 * _np.mean(_abs(b))
        print('thresh = ', thresh)
    thresh2 = 2.0 * thresh

    m1 = (b[1:-1,1:-1]<0) & (b[1:-1,2:  ]>0) & (_abs(b[1:-1,1:-1]-b[1:-1,2:  ])>thresh) # H[- +]
    m2 = (b[1:-1,0:-2]>0) & (b[1:-1,1:-1]<0) & (_abs(b[1:-1,0:-2]-b[1:-1,1:-1])>thresh) # H[+ -]
    m3 = (b[1:-1,1:-1]<0) & (b[2:  ,1:-1]>0) & (_abs(b[1:-1,1:-1]-b[2:  ,1:-1])>thresh) # V[- +]
    m4 = (b[0:-2,1:-1]>0) & (b[1:-1,1:-1]<0) & (_abs(b[0:-2,1:-1]-b[1:-1,1:-1])>thresh) # V[+ -]
    m5 = (b[1:-1,1:-1]==0) & ((b[1:-1,0:-2]*b[1:-1,2:  ])<0) & (_abs(b[1:-1,0:-2]-b[1:-1,2:  ])>thresh2) # H[- 0 +]
    m6 = (b[1:-1,1:-1]==0) & ((b[0:-2,1:-1]*b[2:  ,1:-1])<0) & (_abs(b[0:-2,1:-1]-b[2:  ,1:-1])>thresh2) # V[- 0 +]
    
    m = m1 | m2 | m3 | m4 | m5 | m6
    m = _np.pad(m, ((1,1),(1,1)), mode='constant', constant_values = False)
    return m
   