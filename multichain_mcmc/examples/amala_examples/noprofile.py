import model8

import pymc 
import multichain_mcmc
import numpy 
import time
from pylab import *

start = time.time()
sampler = multichain_mcmc.AmalaSampler(model8.model_gen)
sampler.sample(ndraw = 500,  maxGradient = 1.3, mAccept = True, mConvergence = True)
print  (time.time() - start)    

history = sampler.history
samples = sampler.samples
slices = sampler.slices

print history
subplot(1,2,1)
plot(history[:, slices['mu']])
subplot(1,2,2)
hist(samples[:, slices['mu']])
show()