'''
Created on Nov 24, 2009

@author: johnsalvatier
'''
import model7

import pymc 
import multichain_mcmc
from pylab import *
import numpy 

variables_of_interest = ['mu']

sampler = multichain_mcmc.AmalaSampler(model7.model_gen)
sampler.sample(ndraw = 500,samplesPerAdapatationParameter = .5,adaptationDecayLength = 100, variables_of_interest = variables_of_interest, ndraw_max = 10000, maxGradient = 100, mAccept = True, mConvergence = True)
slices = sampler.slices

print sampler.R
history = sampler.samples
print history.shape
samples = history.shape[0]
print sampler.accepts_ratio
print sampler.burnIn
print "running time (s)", sampler.time 

meanInterceptEstimate = numpy.mean(history[:,slices['intercept']], axis = 0)
meanSdEstimate = numpy.mean(history[:,slices['sd']], axis = 0)
meanResponsesEstimates = numpy.mean(history[:,slices['responses']], axis = 0)


print meanInterceptEstimate, meanSdEstimate
print meanInterceptEstimate/model7.trueIntercept, meanSdEstimate/model7.trueErrorSd
print meanResponsesEstimates
print meanResponsesEstimates/model7.trueResponses


subplot(3,3,1)
hist(history[:, slices['intercept']])
subplot(3,3,2)
hist(history[:, slices['sd']])
subplot(3,3,3)
hist(history[:, 2])
subplot(3,3,4)
hist(history[:, 3])
subplot(3,3,5)
plot(history[:, slices['intercept']])
subplot(3,3,6)
plot(history[:, slices['sd']])
subplot(3,3,7)
plot(history[:, 2])
subplot(3,3,8)
plot(history[:, 3])
subplot(3,3,9)
plot(history[:, 4])

show()


