'''
Created on Oct 24, 2009
http://stackoverflow.com/questions/311703/algorithm-for-sampling-without-replacement
@author: johnsalvatier
'''

from __future__ import division
import numpy 

def random_no_replace(sampleSize, populationSize, numSamples):
    
    samples  = numpy.zeros((numSamples, sampleSize),dtype=int)
    
    # Use Knuth's variable names
    cdef int n = sampleSize
    cdef int N = populationSize

    cdef i = 0
    cdef int t = 0 # total input records dealt with
    cdef int m = 0 # number of items selected so far
    cdef double u

    while i < numSamples:

        t = 0
        m = 0 
        while m < n :

            
            u = numpy.random.uniform() # call a uniform(0,1) random number generator

            if  (N - t)*u >= n - m :
            
                t += 1
            
            else:
            
                samples[i,m] = t
                t += 1
                m += 1
                
        i += 1
        
    return samples
    

