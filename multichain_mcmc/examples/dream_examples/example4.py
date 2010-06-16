'''
Created on Nov 24, 2009

@author: johnsalvatier
'''
import model4

import pymc 
import multichain_mcmc
from pylab import *
import numpy 
#NOT A WORKING EXAMPLE
sampler = multichain_mcmc.DreamSampler(model4.model_gen)
sampler.sample(nChains = 10,adaptationRate = 'auto', mConvergence = True,mAccept = True)
slices = sampler.slices

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
subplot(3,3,3)
hist(history[:, 3])
subplot(3,3,5)
plot(history[:, 0])
subplot(3,3,6)
plot(history[:, 1])
subplot(3,3,8)
plot(history[:, 2])
subplot(3,3,9)
plot(history[:, 3])


print model4.trueFactorMagnitudes
print model4.trueFactorLoadings
print model4.trueErrorSds

meanLoadingEstimates = numpy.mean(history[:,slices['factorloadings']], axis = 0)
meanSdEstimates = numpy.mean(history[:,slices['residualsds']], axis = 0)
print meanSdEstimates 
print meanLoadingEstimates/model4.trueFactorLoadings
print meanSdEstimates/model4.trueErrorSds



show()
