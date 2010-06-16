'''
Created on Nov 24, 2009

@author: johnsalvatier
'''
import model3

import pymc 
import multichain_mcmc
from pylab import *

sampler = multichain_mcmc.DreamSampler(model3.model_gen)
sampler.sample(nChains = 5, ndraw = 500, mConvergence = True,mAccept = True)


print sampler.R
history = sampler.history
print history.shape
samples = history.shape[0]
print sampler.acceptRatio
print sampler.burnIn

subplot(3,4,1)
hist(history[:, 0])
subplot(3,4,2)
hist(history[:, 1])
subplot(3,4,3)
hist(history[:, 2])
subplot(3,4,4)
hist(history[:, 3])
subplot(3,4,5)
plot(history[:, 0])
subplot(3,4,6)
plot(history[:, 1])
subplot(3,4,7)
plot(history[:, 2])
subplot(3,4,8)
plot(history[:, 3])
subplot(3,4,9)
plot(history[:,1],history[:, 2],'.')
subplot(3,4,10)
plot(history[:,2],history[:, 3],'.')
subplot(3,4,11)
plot(history[:,1],history[:, 3],'.')
subplot(3,4,12)
plot(history[:,0],history[:, 3],'.')



show()
