'''
Created on Nov 24, 2009

@author: johnsalvatier
'''
import model2

import pymc 
import multichain_mcmc
from pylab import *

sampler = multichain_mcmc.DreamSampler(model2.model_gen)
sampler.sample(nChains = 5, ndraw = 1000)


print sampler.R
history = sampler.history
print history.shape
samples = history.shape[0]
print sampler.accepts_ratio
print sampler.burnIn
print sampler.time 

subplot(3,2,1)
hist(history[:, 0])
subplot(3,2,2)
hist(history[:, 1])
subplot(3,2,3)
hist(history[:, 2])
subplot(3,2,4)
plot(history[:, 0])
subplot(3,2,5)
plot(history[:, 1])
subplot(3,2,6)
plot(history[:,1],history[:, 2],'.')



show()
