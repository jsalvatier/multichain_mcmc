'''
Created on Jan 27, 2010

@author: johnsalvatier
'''
import numpy  as np

def eigen(a, n = -1):
    
    if len(a.shape) == 0: # if we got a 0-dimensional array we have to turn it back into a 2 dimensional one 
        a = a[np.newaxis,np.newaxis]
        
    if n == -1:
        n = a.shape[0]
        
    eigenvalues, eigenvectors = np.linalg.eigh(a)

    indicies = np.argsort(eigenvalues)[::-1]
    return eigenvalues[indicies[0:n]], eigenvectors[:,indicies[0:n]]

def vectorsMult( matrix, vectors):
    """
    multiply a matrix by a list of vectors which should result in another list of vectors 
    the "list" axis should be the first axis
    """
    return np.dot(matrix, vectors.transpose()[np.newaxis,:,:])[:,0].transpose()


def truncate_gradient(vectors, gradLogPs, adapted_approximation, maxGradient):
    
    transformedGradients = vectorsMult(adapted_approximation.basis, gradLogPs)
    transformedVectors = vectorsMult(adapted_approximation.transformation, vectors - adapted_approximation.location)

    # truncate by rescaling
    normalNorms =np.sum(transformedVectors**2, axis = 1)**.5
    gradientNorms = np.sum(transformedGradients**2, axis = 1)**.5

    truncation = maxGradient * normalNorms
    
    return (truncation/np.maximum(truncation, gradientNorms))[:, np.newaxis] * gradLogPs