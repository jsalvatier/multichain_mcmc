#!/usr/bin/env python
import numpy
from numpy import *
from pymc import *


N = 12


n = 120


i_logvol = random.normal(size = (n, N))
f_logvol = random.normal(size = (n, 1))
data_shape = ones ((n -1,N))

def gen_model():
   
    mean_hyper_mean = Normal('mean_hyper_mean', mu = 0, tau = 1.0/3**2)
    mean_hyper_logvar = Normal('mean_hyper_logvar', mu = -.5, tau = 1.0/1.0**2)
    mean = Normal('mean', mu = mean_hyper_mean * ones(N)[newaxis,:], tau = 1.0/exp(mean_hyper_logvar))

    
    ad_hyper_mean = Normal('ad_hyper_mean', mu = .2 , tau = 1.0/1.0**2)
    ad_hyper_logvar = Normal('ad_hyper_logvar', mu = -.5 , tau = 1.0/1.0**2)
    ad = Normal('ad', mu = ad_hyper_mean * ones(N)[newaxis,:], tau = 1.0/exp(ad_hyper_logvar))
    

    f1d_hyper_mean = Normal('f1d_hyper_mean', mu = .5 , tau = 1.0/2.0**2)
    f1d_hyper_logvar = Normal('f1d_hyper_logvar', mu = -.5 , tau = 1.0/1.0**2)
    f1d = Normal('f1d', mu = f1d_hyper_mean * ones(N)[newaxis,:], tau = 1.0/exp(f1d_hyper_logvar))

    logvar_hyper_mean = Normal('logvar_hyper_mean', mu = -.5 , tau = 1.0/2.0**2)
    logvar_hyper_logvar = Normal('logvar_hyper_logvar', mu = -.5 , tau = 1.0/1.0**2)
    logvar = Normal('logvar', mu =  logvar_hyper_mean * zeros(N)[newaxis,:] , tau =  1.0/exp(logvar_hyper_logvar))
    
    print 1
    tau1 =  exp(logvar)
    print 2, tau1, data_shape, tau1.__array_priority__
    tau2 = 1.0/tau1
    print 3
    z = data_shape/exp(f1d_hyper_logvar)
    print 4
    tau = data_shape * tau2
    print 5
    
    volatilities = Normal('vols', mu = mean + ad * i_logvol[0:(n-1),:] + f1d * f_logvol[1:n,:] , tau = data_shape/exp(logvar) , observed = True, value = i_logvol[1:n,:])
    return [mean, mean_hyper_mean, mean_hyper_logvar,
            ad, ad_hyper_mean, ad_hyper_logvar, 
            f1d,f1d_hyper_mean, f1d_hyper_logvar, 
            logvar, logvar_hyper_mean, logvar_hyper_logvar]