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
    
    def __init__ (self, convergence_criteria, history, group ='all'):
        self.convergence_criteria = convergence_criteria
        self.history = history
        self.group   = group
    
    _R = numpy.Inf
    _V = numpy.Inf
    _VChange = numpy.Inf
    
    _W = numpy.Inf
    _WChange = numpy.Inf
    
    def _get_R(self):
        return self._R
    
    R = property(_get_R)
    
    @property
    def VChange(self):
        return self._VChange
    
    @property
    def WChange(self):
        return self._WChange
    
    def update(self):
        """
        Updates the convergence diagnostic with the current history.
        """
        
        N = self.history.nsequence_histories
        
        sequences = self.history.sequence_histories

        variances  = numpy.var(sequences,axis = 2)

        means = numpy.mean(sequences, axis = 2)
        
        withinChainVariances = numpy.mean(variances, axis = 0)
        
        betweenChainVariances = numpy.var(means, axis = 0) * N
        
        varEstimate = (1 - 1.0/N) * withinChainVariances + (1.0/N) * betweenChainVariances

        self._R = numpy.sqrt(varEstimate/ withinChainVariances)
        
        self._W = withinChainVariances
        self._WChange = numpy.abs(numpy.log(withinChainVariances /self._W)**.5)
        
        self._V = varEstimate
        self._VChange = numpy.abs(numpy.log(varEstimate /self._V)**.5)
        
    def converged(self):
        return all(abs(log(self._R)) < self.convergence_criteria)
    
    def state(self):
        return str(log(self._R))
        
class CovarianceConvergence:

    relative_scales = Inf
    
    def __init__(self, convergence_criteria, history, group = 'all' ):
        self.history = history
        self.group   = group
        self.convergence_criteria = convergence_criteria
    
    def update(self):
        
        relevant_history = self.history.group_combined_history(self.group)
        
        end = relevant_history.shape[0]
        midpoint = floor(end/2)
        
        covariance1 = numpy.cov(relevant_history[0:midpoint, :].transpose())
        covariance2 = numpy.cov(relevant_history[midpoint:end, :].transpose())

        eigenvalues1, eigenvectors1 = eigen(covariance1)
        basis1 = (sqrt(eigenvalues1)[newaxis,:] * eigenvectors1)
        
        eigenvalues2, eigenvectors2 = eigen(covariance2)
        basis2 = (sqrt(eigenvalues2)[newaxis,:] * eigenvectors2)  
        
        # project the second basis onto the first basis
        projection = dot(linalg.inv(basis1), basis2)
        
        # find the releative size in each of the basis1 directions 
        self.relative_scales = log(sum(projection**2, axis = 0)**.5) 
    
    def converged(self):
        return all(abs(self.relative_scales) < self.convergence_criteria)
    
    def state(self):
        return "relative scales: " + str(self.relative_scales)
    
