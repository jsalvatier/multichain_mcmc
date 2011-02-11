'''
Created on Feb 2, 2011

@author: jsalvatier
'''
import pylab as pyl
from pylab import hist, plot
import numpy as np

id = lambda x : x 

def slice_len(s):
    return s.stop - s.start


def hst(sampler,name, n = 0):
    return sampler.history[:,sampler.slices[name]][:,n]
def smp(sampler,name, n = 0):
    return sampler.samples[:,sampler.slices[name]][:,n]


def show_samples(plot_func, sampler, names, transformations = {} , *args):
    
    slices = sampler.slices
    
    N_plots  = sum(slice_len(slices[name]) for name in names)
    
    rowcol = np.ceil(np.sqrt(N_plots))
    
    plot_num = 1
    
    pyl.figure()
    for name in names:   
        try :
            transformation = transformations[name]
        except KeyError : 
            transformation = id
        for i in range(slice_len(slices[name])):
            
            pyl.subplot(rowcol, rowcol, plot_num)
            plot_func(transformation(sampler.samples[:,slices[name]][:,i]), *args) 
            pyl.title(name + str(i))
            
            plot_num += 1
        
    pyl.show()