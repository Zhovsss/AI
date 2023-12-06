"""
Custom SVM Kernels

Author: Eric Eaton, 2014

"""

import numpy as np
from numpy import linalg as LA

_polyDegree = 2
_gaussSigma = 1


def myPolynomialKernel(X1, X2):
    '''
        Arguments:  
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    
    return (1+np.dot(X1,X2.T))**_polyDegree



def myGaussianKernel(X1, X2):
    '''
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    
    n1,d1 = X1.shape
    n2,d2 = X2.shape

    # find the distance between the v and w vectors to be used in the 
    # K(v,w) equation
    distance = np.zeros((n1, n2))
    for i in range(0, n2):
      distance[:,i] = np.sum((X1 - X2[i,:]) ** 2, axis = 1)

    # compute the K(v,w) equation

    sig = 2 * (_gaussSigma ** 2)
    return np.exp(-distance / sig)



def myCosineSimilarityKernel(X1,X2):
    '''
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    
    # X, Y = check_pairwise_arrays(X, Y)

    # X1_normalized = LA.norm(X1)
    # if X1 is X2:
    #     X2_normalized = LA.norm(X1)
    # else:
    #     X2_normalized = LA.norm(X2)

    # norm = X1_normalized * X2_normalized.T
    # return np.dot(X1, X2.T)/norm
    n1,d1 = X1.shape
    n2,d2 = X2.shape
    
    
    # prod = np.dot(X1,X2.T)
    norm_1 = np.sqrt((X1 ** 2).sum(axis=1)).reshape(X1.shape[0], 1)
    norm_2 = np.sqrt((X2 ** 2).sum(axis=1)).reshape(X2.shape[0], 1)
    return X1.dot(X2.T) / (norm_1 * norm_2.T)

    # find the distance between the v and w vectors to be used in the 
    # K(v,w) equation
    # s1 = LA.norm(X1)
    # s2 = LA.norm(X2)
    # norm = s1*s2
    # return np.dot(X1,X2.T)/norm

    # return K


    # X1=X1.toarray()
    # X2=X2.toarray()
    # X1=np.array(X1.todense())
    # X2=np.array(X2.todense())
    # norm = LA.norm(X1) * LA.norm(X2)
    # return np.dot(X1, X2.T)/norm


