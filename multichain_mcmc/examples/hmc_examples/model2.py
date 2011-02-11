'''
Created on Nov 24, 2009

@author: johnsalvatier
'''
from numpy import *
from pymc import *

n = 2
observations = random.normal(size = n)
tau = array([1,10.0])**-2

def model():
    
    
    variables = []
    mean = Normal("mean", mu = zeros(n), tau = 1.0/(2.0**2))
    variables.append(mean)

    x = Normal("x",mu = sum(mean), tau = tau, observed= True, value = observations)
    variables.append(x)
    
    return variables
