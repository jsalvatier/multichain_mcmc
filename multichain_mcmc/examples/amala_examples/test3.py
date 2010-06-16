'''
Created on Jan 20, 2010

@author: johnsalvatier
'''

from numpy import *
from pymc import *
import pymc

class testClass:
    
    __array_priority__ = 1000
    

        
    def __index__(self):
        print "index"
        
    def __rmul__(self, other):
        print "rmul"
        return 1
  
# this is the way that pymc does this   
def __getitem__(self, index):
    print "getitem", index   
    
testClass.__getitem__ = types.UnboundMethodType(__getitem__, None, testClass)

x = array([[1,2,3,7],[4,5,1,0]])
y = testClass()


a = x * y