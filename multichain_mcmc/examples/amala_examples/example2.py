'''
Created on Nov 24, 2009

@author: johnsalvatier
'''
import model2

import pymc 
import multichain_mcmc
from pylab import *

sampler = multichain_mcmc.AmalaSampler(model2.model)
sampler.sample(nChains = 5,ndraw = 500,samplesPerAdapatationParameter = .5,adaptationDecayLength = 100,  thin = 5)


print sampler.R
history = sampler.history
slices = sampler.slices
print history.shape
samples = history.shape[0]
print sampler.accepts_ratio
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
