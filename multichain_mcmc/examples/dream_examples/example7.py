'''
Created on Nov 24, 2009

@author: johnsalvatier
'''
import model7

import pymc 
import multichain_mcmc
from pylab import *
import numpy 

sampler = multichain_mcmc.DreamSampler(model7.model_gen)
sampler.sample(nChains = 10, ndraw = 1000, adaptationRate = 'auto', mAccept = True, mConvergence = True)
slices = sampler.slices

print sampler.R
history = sampler.history
print history.shape
samples = history.shape[0]
print sampler.accepts_ratio
print sampler.burnIn

meanInterceptEstimate = numpy.mean(history[:,slices['intercept']], axis = 0)
meanSdEstimate = numpy.mean(history[:,slices['sd']], axis = 0)
meanResponsesEstimates = numpy.mean(history[:,slices['responses']], axis = 0)


print meanInterceptEstimate, meanSdEstimate
print meanInterceptEstimate/model7.trueIntercept, meanSdEstimate/model7.trueErrorSd
print meanResponsesEstimates
print meanResponsesEstimates/model7.trueResponses


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


