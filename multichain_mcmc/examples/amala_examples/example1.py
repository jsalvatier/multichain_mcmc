'''
Created on Nov 24, 2009

@author: johnsalvatier
'''
import model1

import pymc 
import multichain_mcmc
from pylab import *

sampler = multichain_mcmc.AmalaSampler(model1.model)
sampler.sample(nChains = 5, ndraw = 500,  maxGradient = 100)


print sampler.R
history = sampler.history
slices = sampler.slices
print history.shape
samples = history.shape[0]
print sampler.accepts_ratio
print sampler.burnIn
print sampler.time 

subplot(3,3,1)
hist(history[:, slices['a']])
subplot(3,3,2)
hist(history[:, slices['b']])
subplot(3,3,3)
hist(history[:, slices['sd']])

subplot(3,3,4)
plot(history[:, slices['a']])
subplot(3,3,5)
plot(history[:, slices['b']])
subplot(3,3,6)
plot(history[:, slices['sd']])

subplot(3,3,7)
plot(history[:, slices['a']], history[:, slices['b']], '.')

show()
