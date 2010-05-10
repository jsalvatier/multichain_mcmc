import model7

import pymc 
import multichain_mcmc
from pylab import *
import numpy 
import cProfile 

def sample():
    sampler = multichain_mcmc.AmalaSampler(model7.model_gen)
    sampler.sample(ndraw = 500,  maxGradient = 1.3, mAccept = True, mConvergence = True)
    

cProfile.run("sample()")