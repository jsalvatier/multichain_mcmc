'''
Created on Nov 24, 2009

@author: johnsalvatier
'''
import model2

import pymc 
import multichain_mcmc as mc
from pylab import *

import pydevd
pydevd.set_pm_excepthook()
import numpy 
numpy.seterr(all = 'raise')
sampler = mc.HMCSampler(model2.model)
history, time  = sampler.sample(nChains = 5, ndraw = 3000,  maxGradient = 100)

print time
mc.show_samples(plot, history, ('mean', 'sd'))    
