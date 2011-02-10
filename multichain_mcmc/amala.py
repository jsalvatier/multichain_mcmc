'''
Created on Jan 13, 2010

@author: johnsalvatier                 
'''
from __future__ import division
from pymc import *

from numpy import *
from convergence import GRConvergence, CovarianceConvergence
from adaptation import AdaptedApproximation, AdaptedScale
import scipy.optimize
from multichain import MultiChainSampler, MultiChain
import time 
import dream_components
from utilities import vectorsMult, eigen,truncate_gradient
from simulation_history import *
from mode_finding import find_mode

class AmalaSampler(MultiChainSampler):
    """
    Implementation of Adaptive Metropolist adjusted Langevin algorithm (Adaptive MaLa) MCMC sampling technique.
    Requires PyMC branch with gradient information support in order to function:
        http://github.com/pymc-devs/pymc/tree/gradientBranch
    
    MaLa uses gradient information in order to bias the proposal distribution in the direction of increasing 
    probability density. Adaptive technique based on (1). Multiple chains are used in order to attempt to 
    improve convergence monitoring and adaptation. Resampling from the past (2) is used in order to avoid
    getting stuck in low probability regions. However (2) shows that resampling from the past generally slows
    down sampling, so we try to do this mostly near the beginning of sampling. Convergence assessment is 
    based on a naive implementation of the Gelman-Rubin convergence statistics.
    
    (1) Atchade, Y. (2006). An Adaptive Version for the Metropolis Adjusted Langevin Algorithm with a Truncated Drift. Methodology and Computing in Applied Probability. 8 (2), 235-254. 
    (2) Atchade, Y. (2009). Resampling from the past to improve on MCMC algorithms. FAR EAST JOURNAL OF THEORETICAL STATISTICS. 27 (1), 81-100. 
       
    """
    optimalAcceptance = .574
    A1 = 1e7
    e1 = 1e-5
    e2 = 1e-5
    outliersFound = 0
    
    
    acceptRatio = 0.0
    
    def sample(self, ndraw = 1000, samplesPerAdapatationParameter = 3, adaptationDecayLength = 250, variables_of_interest = None,minimum_scale = .1, maxGradient = 1.0, ndraw_max = None , nChains = 5, burnIn = 1000, thin = 2, initial_point = None, convergenceCriteria = 1.1, monitor_convergence = True, monitor_acceptence = True):
        """Samples from a posterior distribution using Adaptive Metropolis Adjusted Langevin Algorithm (AMALA).
        
        Parameters
        ----------
        ndraw : int 
            minimum number of draws from the sample distribution to be returned 
        ndraw_max : int 
            maximum number of draws from the sample distribution to be returned
        nChains : int 
            number of different chains to employ
        burnInSize : int
            number of iterations (meaning draws / nChains) to do before doing actual sampling.
        minimum_scale : float
            the minimum that the scaling constant can fall to (default .1)
        monitor_convergence : bool
            determines whether to periodically print out convergence statistics (True)
        monitor_acceptence : bool
            determines whether to periodically print out the average acceptance ratio and adapted scale 
        initial_point : dictionary  
            
        Returns
        -------
            None : None 
                sample sets 
                self.history which contains the combined draws for all the chains
                self.iter which is the total number of iterations 
                self.acceptRatio which is the acceptance ratio
                self.burnIn which is the number of burn in iterations done 
                self.R  which is the gelman rubin convergence diagnostic for each dimension
        """
        startTime = time.time()
        if ndraw_max is None:
            ndraw_max = 10 * ndraw
        
        maxChainDraws = floor(ndraw_max/nChains)        
        self._initChains(nChains, ndraw_max)   

        history = SimulationHistory(maxChainDraws,self._nChains, self.dimensions)
        
        if variables_of_interest is not None:
            slices = []
            for var in variables_of_interest:
                slices.append(self.slices[var])
        else:
            slices = [slice(None,None)]
            
            
        history.add_group('interest', slices)


        # initilize the convergence diagnostic object
        
        convergence_diagnostics = [GRConvergence(.1, history),
                                   CovarianceConvergence(.3, history),
                                   CovarianceConvergence(.2, history, 'interest')]
        
        monitor_diagnostics = [convergence_diagnostics[0], convergence_diagnostics[2]]     
        
        iter = 1

        lastRecalculation = 0
               
        # try to find some approximate modes for starting the chain 
        for chain in self._chains:
            inv_hessian = find_mode(self,chain)    

        adaptationConstant = min(self._nChains/(self.dimensions * samplesPerAdapatationParameter), 1)

        adapted_approximation = AdaptedApproximation(self._chains[1].vector, inv_hessian)
        adapted_scale = AdaptedScale(self.optimalAcceptance, minimum_scale)

        accepts_ratio_weighting = 1 - exp(-1.0/30) 
        adaptationDecay = 1.0/adaptationDecayLength

        # continue sampling if:
        # 1) we have not drawn enough samples to satisfy the minimum number of iterations
        # 2) or any of the dimensions have not converged 
        # 3) and we have not done more than the maximum number of iterations 

        while ( (history.nsamples < ndraw or 
                not all((diagnostic.converged() for diagnostic in convergence_diagnostics))) and 
                history.ncomplete_sequence_histories < maxChainDraws - 1):

            if iter  == burnIn:
                history.start_sampling()

            current_logps = self.logps

            jump_logp, reverse_logp = propose_amala(self,adapted_approximation, adapted_scale, maxGradient)
   
            acceptance = self.metropolis_hastings(current_logps,self.logps, jump_logp, reverse_logp) 
                
            self._update_accepts_ratio(accepts_ratio_weighting, acceptance)
            
            if monitor_acceptence and iter % 20 == 0:
                print "accepts ratio: ", self.accepts_ratio, " adapted scale: ", adapted_scale.scale
      
      
            if history.nsamples > ndraw and history.nsamples > lastRecalculation * 1.1:

                lastRecalculation = history.nsamples
                
                for diagnostic in convergence_diagnostics:
                    diagnostic.update()
                
                for diagnostic in monitor_diagnostics:
                    print diagnostic.state()


            if iter % thin == 0:
                history.record(self.vectors, self.logps, .5)
            
            adaptation_rate = adaptationConstant * exp(-history.nsamples * adaptationDecay)
            adapted_approximation.update(self.vectors, adaptation_rate)
            adapted_scale.update(acceptance, adaptation_rate)

            iter += 1

        self.finalize_chains()
        
        return history , time.time() - startTime
            
    
def propose_amala(chains, adapted_approximation, adapted_scale, maxGradient):
    
    scaledOrientation = adapted_approximation.orientation * adapted_scale.scale**2 
    
    def drift(x, gradient):
        return vectorsMult(scaledOrientation, truncate_gradient(x, gradient,adapted_approximation, maxGradient) /2)
    
    def jump_logp (jump):
        return array([pymc.distributions.mv_normal_cov_like(x = jump[i,:], mu = zeros(chains.dimensions), C = scaledOrientation) for i in range(jump.shape[0])])
        
    
    
    forward_jump = (drift(chains.vectors, chains.logp_grads) +
                 random.multivariate_normal(mean = zeros(chains.dimensions) ,cov = scaledOrientation, size = chains._nChains))
    
    
    chains.propose(chains.vectors + forward_jump)
    
    backward_jump = - forward_jump - drift(chains.vectors, chains.logp_grads)
    
    return jump_logp(forward_jump), jump_logp(backward_jump)
    