from __future__ import division
from numpy import *
import numpy 
import math
from utilities import *

'''
Created on Jan 11, 2010

@author: johnsalvatier
'''

class GRConvergence:
    """
    Gelman Rubin convergence diagnostic calculator class. It currently only calculates the naive
    version found in the first paper. It does not check to see whether the variances have been
    stabilizing so it may be misleading sometimes.
    """
    _R = numpy.Inf
    _V = numpy.Inf
    _VChange = numpy.Inf
    
    _W = numpy.Inf
    _WChange = numpy.Inf
    
    def __init__(self):
        pass
    
    def _get_R(self):
        return self._R
    
    R = property(_get_R)
    
    @property
    def VChange(self):
        return self._VChange
    
    @property
    def WChange(self):
        return self._WChange
    
    def update(self, history):
        """
        Updates the convergence diagnostic with the current history.
        """
        
        N = history.nsequence_histories
        
        sequences = history.sequence_histories

        variances  = numpy.var(sequences,axis = 2)

        means = numpy.mean(sequences, axis = 2)
        
        withinChainVariances = numpy.mean(variances, axis = 0)
        
        betweenChainVariances = numpy.var(means, axis = 0) * N
        
        varEstimate = (1 - 1.0/N) * withinChainVariances + (1.0/N) * betweenChainVariances

        self._R = numpy.sqrt(varEstimate/ withinChainVariances)
        
        self._WChange = numpy.abs(numpy.log(withinChainVariances /self._W)**.5)
        self._W = withinChainVariances
        
        self._VChange = numpy.abs(numpy.log(varEstimate /self._V)**.5)
        self._V = varEstimate
        
class CovarianceConvergence:
    """
    """
    relativeVariances = {}
    
    def update(self, history, group):
        
        relevantHistory = history.group_combined_history(group)


        self.relativeVariances[group] = self.rv(relevantHistory)
     
    @staticmethod   
    def rv(relevantHistory):  
        end = relevantHistory.shape[0]
        midpoint = floor(end/2)
        
        covariance1 = numpy.cov(relevantHistory[0:midpoint, :].transpose())
        covariance2 = numpy.cov(relevantHistory[midpoint:end, :].transpose())

        eigenvalues1, eigenvectors1 = eigen(covariance1)
        basis1 = (sqrt(eigenvalues1)[newaxis,:] * eigenvectors1)
        
        eigenvalues2, eigenvectors2 = eigen(covariance2)
        basis2 = (sqrt(eigenvalues2)[newaxis,:] * eigenvectors2)  
        
        # project the second basis onto the first basis
        projection = dot(linalg.inv(basis1), basis2)
        
        # find the releative size in each of the basis1 directions 
        return log(sum(projection**2, axis = 0)**.5) 
    
    @staticmethod
    def compare_cov(d, cov1, n1, cov2, n2):
        n = n1 + n2
        k = 2
        pooledWithinCov = (cov1*(n1 - 1) + cov2*(n2 -1))/(n - k)
        
        M = (n - k) * log (det(pooledWithinCov))  - (n1 - 1) * log(det(cov1)) - (n2 - 1) * log(det(cov2))
        h = 1 - (2 * d**2 + 3*d - 1)/(6 *(p+1)*(k-1)) * ( 1/(n1 - 1) + 1/(n2 -1)     - 1/(n - k))
        df = d * (d+1) *(k-1) /2
        
        return df, M * h