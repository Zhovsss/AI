import numpy as np
def mapFeature(x1, x2):
    '''
    Maps the two input features to quadratic features.
        
    Returns a new feature array with d features, comprising of
        X1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, ... up to the 6th power polynomial
        
    Arguments:
        X1 is an n-by-1 column matrix
        X2 is an n-by-1 column matrix
    Returns:
        an n-by-d matrix, where each row represents the new features of the corresponding instance
    '''
    
    
    n = x1.size
    arr = np.zeros((n,28))
    for row in range(n):
        count = 0
        for i in range (7):
            for j in range (i+1):
                exp2 = j
                exp1 = i-j
                arr[row][count] = (x1[row]**exp1) * (x2[row]**exp2)
                count += 1
    return arr







    # arr = []
    # for i in range(0, x1.size):
    #     cur = []
        
    #     for k in range(0, x2.size):
    #         for j in range(0, 7):
    #             for n in range(0,7-j):
    #                 if j == 0 and n == 0:
    #                     continue
    #                 cur.append((x1[i]**j)*(x2[k]**n))
    #     arr.append(cur)
    # arr = np.array(arr)
    # print(arr.shape)
    # arr = np.c_[np.ones((arr.shape[0],1)), arr]
    # # print(arr)
    
    # return arr
