'''
Created on Jan 20, 2010

@author: johnsalvatier
'''

from numpy import *
from pymc import *
import pymc

a = ones((3,4))

j= NumpyDeterministics.sum_jacobian('a', a = a, axis = 0)

print j * 1

x = Normal('z', mu = ones((3,4)), tau = ones((3,4)))
y = pymc.sum(x, axis = 1)

print y
