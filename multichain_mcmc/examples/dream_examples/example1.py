'''
Created on Nov 24, 2009

@author: johnsalvatier
'''
import model

import pymc 
import multichain_mcmc
from pylab import *

sampler = multichain_mcmc.DreamSampler(model.model)
sampler.sample()


print sampler.iter 
history = sampler.history
samples = history.shape[0]
print samples

subplot(2,1,1)
hist(history[:, 0])
subplot(2,1,2)
hist(history[:, 1])
show()
