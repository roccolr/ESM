from numpy import array, min, max

def fshs(x:array, k:int)->array:
    x_min = min(x)
    x_max = max(x)
    return ((k-1)*(x-x_min)/(x_max-x_min))