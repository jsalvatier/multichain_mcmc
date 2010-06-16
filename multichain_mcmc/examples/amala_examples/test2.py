'''
Created on Jan 20, 2010

@author: johnsalvatier
'''

from numpy import *
from pymc import *
import pymc
from scipy.stats import distributions as d

print "start"
nPredictors = 20
observations = 100

def rmul2 (self, other):
    print "rmul2"
    return 0

observedPredictors = d.norm(loc = 0, scale = 1.0).rvs((nPredictors,observations))

responses = pymc.Normal("responses", mu = zeros((nPredictors,1)), tau = ones((nPredictors,1)) * 5**-2 )

print 1
print responses.__rmul__
responses.__rmul__ = rmul2
o =  observedPredictors * responses