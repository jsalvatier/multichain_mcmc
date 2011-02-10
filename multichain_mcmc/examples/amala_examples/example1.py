'''
Created on Nov 24, 2009

@author: johnsalvatier
'''

import model1

import pymc 
import multichain_mcmc as mc
from pylab import *
import numpy

import pydevd
pydevd.set_pm_excepthook()

sampler = mc.AmalaSampler(model1.model)
history, time  = sampler.sample(nChains = 5, ndraw = 3000,  maxGradient = 100)

print time
mc.show_samples(plot, history, ('a','b', 'sd'))    