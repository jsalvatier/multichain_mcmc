'''
Created on Nov 24, 2009

@author: johnsalvatier
'''
import model2

import pymc 
import multichain_mcmc
from pylab import *

sampler = multichain_mcmc.HamiltonianSampler(model2.model)
sampler.sample(nChains = 5, ndraw = 300,adaptationConstant = .5, steps = 10, burnIn = 20, debug = 20, mConvergence = True,mAccept = True)


print sampler.R
history = sampler.history
slices = sampler.slices
print history.shape
samples = history.shape[0]
print sampler.acceptRatio
print sampler.burnIn
print sampler.time 

subplot(3,2,1)
hist(history[:, slices['mean']])
subplot(3,2,2)
hist(history[:, slices['sd']])

subplot(3,2,4)
plot(history[:, slices['mean']])
subplot(3,2,5)
plot(history[:, slices['sd']])




show()
