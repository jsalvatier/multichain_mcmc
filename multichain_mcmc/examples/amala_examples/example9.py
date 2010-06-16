'''
Created on Nov 24, 2009

@author: johnsalvatier
'''
import model9

import pymc 
import multichain_mcmc
from pylab import *
from numpy import *
from cPickle import *

sampler = multichain_mcmc.AmalaSampler(model9.gen_model)
sampler.sample(nChains = 5, ndraw = 500,  maxGradient = 100,  mConvergence = True,mAccept = True)


print sampler.R
history = sampler.samples
slices = sampler.slices
print history.shape
samples = sampler.samples
print sampler.accepts_ratio
print sampler.burnIn
print sampler.time 


outfile = open('samples1.pkl', 'wb')
dump(samples, outfile)

outfile = open('slices.pkl', 'wb')
dump(slices, outfile)

for var, slice in slices.iteritems():
    print var, mean(samples[:,slice], axis = 0)
    

"""
subplot(2,3,1)
hist(history[:, slices['ad']])
subplot(2,3,2)
hist(history[:, slices['f1d']])
subplot(2,3,3)
hist(history[:, slices['logvar']])

subplot(2,3,4)
plot(history[:, slices['ad']])
subplot(2,3,5)
plot(history[:, slices['f1d']])
subplot(2,3,6)
plot(history[:, slices['logvar']])
"""


#show()
