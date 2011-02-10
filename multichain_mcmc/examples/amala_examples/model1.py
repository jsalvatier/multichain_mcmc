'''
Created on Nov 25, 2009

@author: johnsalvatier
'''

from numpy import *
import pymc
from scipy import stats
import pylab 
ReData = arange(200, 3000, 25)
measured = 10.2 * (ReData )** .5 + stats.distributions.norm(mu = 0, scale = 55).rvs(len(ReData))
#pylab.plot(ReData, measured)
#pylab.plot(ReData, 1.59e-2 * (ReData )** 1.394,'r' )
#pylab.show()

def model():
    
    varlist = []
    
    sd =pymc.Uniform('sd', lower = 5, upper = 100, value = 55.0) #pymc.Gamma("sd", 60 , beta =  2.0)
    varlist.append(sd)
    
    
    

    
    
    a = pymc.Uniform('a', lower = 0, upper = 100, value = 10.0)#pymc.Normal('a', mu =  10, tau = 5**-2)
    b = pymc.Uniform('b', lower = .05, upper = 2.0, value = .5)
    varlist.append(a)
    varlist.append(b)

    nonlinear = a  * (ReData )** b
    precision = sd **-2

    results = pymc.Normal('results', mu = nonlinear, tau = precision, value = measured, observed = True)
    varlist.append(results)
    
    return varlist