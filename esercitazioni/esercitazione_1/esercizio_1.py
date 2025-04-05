import numpy as np 

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

if __name__ == '__main__':
    x = np.array([-10, 3, -6, 0, 1, -2, 3, 4, -15, 3, 21])
    print(f"Array in ingresso:\t{x}")
    y = clip(x, 8)
    print(f"Array in uscita:\t{y}")

