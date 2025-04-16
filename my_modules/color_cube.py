# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 19:25:33 2020

@author: Davide
"""

import numpy as _np

def draw_color_cube(ax=None, size = 8, shade=False):
    """
    Disegna il cubo dei colori.

    Parameters
    ----------
    ax : Axes3D
    size : Numero di voxel per lato. The default is 8.

    """
    if ax is None:
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        plt.figure()
        ax = plt.axes(projection='3d')
        #ax = Axes3D(); # crea una figura per i grafici 3d
    
    voxels = _np.ones((size,size,size), dtype=_np.float32) # voxel del cubo
    r = _np.arange(-1/(2*size), 1+1/(2*size), 1/size)
    x,y,z = _np.meshgrid(r,r,r, indexing='ij') # coordinate dei voxels
    
    r = _np.arange(0,size) / (size-1)
    colors = _np.stack(_np.meshgrid(r,r,r, indexing='ij'),-1) # colore dei voxels
    
    help(ax.voxels)
    ax.voxels(x,y,z, filled=voxels, facecolors=colors, edgecolor=None, shade=shade) # disegna il cubo


if __name__=="__main__":
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    
    ax = Axes3D(plt.figure(), auto_add_to_figure=True); # crea una figura per i grafici 3d
    
    draw_color_cube(ax, 16)
    ax.set_xlabel('RED')
    ax.set_ylabel('GREEN')
    ax.set_zlabel('BLUE')
    ax.view_init(45,-135)
    plt.show()

    

