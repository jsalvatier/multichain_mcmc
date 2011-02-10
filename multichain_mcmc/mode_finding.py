from numpy import empty, ravel, zeros
from scipy.optimize import fmin_bfgs
from pymc import ZeroProbability

def find_mode(container, chain):
          
    def logp(x):
        chain.propose(x) 
        try:
            return  -chain.logp   
        except ZeroProbability:
            return 300e100
        
    zero = zeros(container.dimensions)
    def grad_logp(x):
        chain.propose(x)
        
        try:
            chain.logp 
        except ZeroProbability:
            return zero
        
        gradient = empty(container.dimensions)
        
        for p, v in chain.logp_gradient.iteritems():
            gradient[container.slices[str(p)]] = ravel(v)
    
        return -gradient
    
    results = fmin_bfgs(logp, chain.vector, grad_logp, disp = False, full_output = True)
    chain.propose(results[0])
    
    return results[3] #inverse hessian