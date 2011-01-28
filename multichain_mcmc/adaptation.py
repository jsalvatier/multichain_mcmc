from __future__ import division
from numpy import *
import numpy 
from utilities import eigen


class AdaptedApproximation:
    
    location = None
    orientation = None
    
    
    transformation = None 
    basis = None

    A1 = 1e7
    e1 = 0
    
    def __init__(self, initial_location, initial_orientation):
        
        initial_orientation = atleast_2d(initial_orientation)
        self.location = initial_location
        self._orientation = initial_orientation
        
        self._n = shape(initial_orientation)[0]
        self._Ie = eye(self._n) * self.e1
        
        self._update_orientation(initial_orientation)
        
        

    def update(self, currentVectors, adaptation_rate):
        
        oldMean = self.location 
        oldOrientation = self._orientation
        
        self.location = project(oldMean + adaptation_rate *  mean(currentVectors - oldMean, axis = 0), self.A1)
        
        self._orientation = project(oldOrientation + adaptation_rate * (dot((currentVectors - oldMean).transpose(),currentVectors - oldMean)/self._n - oldOrientation), self.A1 )
            
        self._update_orientation(self._orientation)
        
        
    _last_calc = Inf
    def _update_orientation(self, orientation):
        orientation = orientation + self._Ie  
        
        if self._last_calc > 20:
            eigenvalues, eigenvectors = eigen(orientation)
              
            # find the basis that will be uncorrelated using the covariance matrix
            basis = (sqrt(eigenvalues)[newaxis,:] * eigenvectors).transpose()
            #find the matrix that will transform a vector into that basis
            
            self.basis = basis
            self.transformation = linalg.inv(basis )
            self._last_calc = 0
        else :
            self._last_calc += 1

        self.orientation = orientation
        
        
class AdaptedScale:
    
    scale = 1.0
    
    
    A1 = 1e7
    
    def __init__(self, optimal_acceptence, minimum_scaling):
        self._optimal_acceptence = optimal_acceptence
        self._minimum_scaling = minimum_scaling 
        
    def update(self, acceptance, adaptation_rate):
            
        self.scale = max(project( self.scale *( 1  + adaptation_rate * mean (acceptance - self._optimal_acceptence)), self.A1), self._minimum_scaling)
        
        
def project(x , A1):
    
    norm = sqrt(sum(x**2))
    if norm < A1:
        return x
    else:
        return x * A1 / norm