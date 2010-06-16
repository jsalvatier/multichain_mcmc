'''
Created on Nov 24, 2009

@author: johnsalvatier
'''
import model6

import pymc 
import multichain_mcmc
from pylab import *
import numpy 

sampler = multichain_mcmc.DreamSampler(model6.model_gen)
sampler.sample( nChains = 5, adaptationRate = 'auto', mConvergence = True,mAccept = True)


print sampler.R
history = sampler.history
print history.shape
samples = history.shape[0]
print sampler.acceptRatio
print sampler.burnIn

subplot(3,3,1)
hist(history[:, 0])
subplot(3,3,2)
hist(history[:, 1])
subplot(3,3,3)
hist(history[:, 2])
subplot(3,3,4)
hist(history[:, 3])
subplot(3,3,5)
plot(history[:, 4])
subplot(3,3,6)
plot(history[:, 5])
subplot(3,3,7)
plot(history[:, 1])
subplot(3,3,8)
plot(history[:, 2])
subplot(3,3,9)
plot(history[:, 3])

show()
