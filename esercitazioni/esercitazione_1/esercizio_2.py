import numpy as np 
import random

def clip(x:np.array) -> np.array:
    """
    Given a 16x16 matrix, returns a matrix containing the 4 greatest values
    """
    N = x.shape[0]
    M = x.shape[1]
    y = np.reshape(x,M*N)
    y = np.sort(y)
    greatest_values = y[-4:]
    # print(greatest_values)
    trashold = np.min((greatest_values))
    print(f'Trashold set:\t{trashold}')

    for i in range(0, N):
        for j in range(0, M):
            if x[i,j] < trashold:
                x[i,j] = 0
    
    return x

def create_random_matrix(dim:tuple) -> np.array:
    N = dim[0]
    M = dim[1]

    elements = []
    for i in range(0, N):
        sublist = []
        for j in range(0, M):
            sublist.append(int(random.random()*100))
        elements.append(sublist)
    
    return np.array(elements)


if __name__ == '__main__':
    x = create_random_matrix((4,4))
    print(x)
    print("Elaboration...")
    y = clip(x)
    print(y)
    

