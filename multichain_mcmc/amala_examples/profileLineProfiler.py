import model8

import pymc 
import multichain_mcmc
import numpy 
import line_profiler
import time 

def sample():
    sampler = multichain_mcmc.AmalaSampler(model8.model_gen)
    sampler.sample(ndraw = 500,  maxGradient = 1.3, mAccept = True, mConvergence = True)
    
start = time.time()

    
    
profile = line_profiler.LineProfiler(pymc.Stochastic.gradient,pymc.Stochastic.grad_logp )
profile.runcall(sample)
profile.print_stats()
print  (time.time() - start)    