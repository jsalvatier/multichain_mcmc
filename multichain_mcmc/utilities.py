'''
Created on Jan 27, 2010

@author: johnsalvatier
'''
import numpy 

def eigen(a, n = -1):
    
    if len(a.shape) == 0: # if we got a 0-dimensional array we have to turn it back into a 2 dimensional one 
        a = a[numpy.newaxis,numpy.newaxis]
        
    if n == -1:
        n = a.shape[0]
        
    eigenvalues, eigenvectors = numpy.linalg.eigh(a)

    indicies = numpy.argsort(eigenvalues)[::-1]
    return eigenvalues[indicies[0:n]], eigenvectors[:,indicies[0:n]]

def vectorsMult( matrix, vectors):
    """
    multiply a matrix by a list of vectors which should result in another list of vectors 
    the "list" axis should be the first axis
    """
    return numpy.dot(matrix, vectors.transpose()[numpy.newaxis,:,:])[:,0].transpose()