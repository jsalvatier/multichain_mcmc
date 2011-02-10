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
from utilities import vectorsMult, eigen 
from simulation_history import *

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
        
        # get the starting log likelihood and position for each of the chains 
        #currentVectors = self._vectors

 
        #initialize the history arrays  
        history = SimulationHistory(maxChainDraws,self._nChains, self.dimensions)
        
        if variables_of_interest is not None:
            slices = []
            for var in variables_of_interest:
                slices.append(self.slices[var])
        else:
            slices = [slice(None,None)]
        history.add_group('interest', slices)
        
        seedChainDraws = int(ceil(self.dimensions* 2/nChains) * nChains)
        history.record(self.draw_from_priors(seedChainDraws), zeros((self._nChains, seedChainDraws)), 0)

        # initilize the convergence diagnostic object
        grConvergence = GRConvergence()
        covConvergence = CovarianceConvergence()

        
        #2)now loop through and sample 
        
        minDrawIters = ceil(ndraw / nChains)
        maxIters = ceil(ndraw_max /nChains)
        
        
        iter = 1
    
        proposalVectors = 0
        reverseJumpLogPs = 0
        jumpLogPs = 0
        proposalGradLogPs = 0
       
        lastRecalculation = 0
        
        
        # try to find some approximate modes for starting the chain 
        for i in range(self._nChains):
            x0 = self._chains[i].vector
            def logp(vector):
                self._chains[i].propose(vector)
                
                try:
                    return  -self._chains[i].logp
                    
                except ZeroProbability:
                    return 300e100
            
            def grad_logp(vector):
                self._chains[i].propose(vector)
                
                try:
                    self._chains[i].logp 
                except ZeroProbability:

                    return zeros(self.dimensions)
                
                gradientd = self._chains[i].logp_gradient
                gradient = empty(self.dimensions)
                
                for p, v in gradientd.iteritems():
                    gradient[self.slices[str(p)]] = -ravel(v)
                
                return gradient
            
            min_params = scipy.optimize.fmin_bfgs(logp, x0, grad_logp, full_output = True, disp = True)
            self._chains[i].propose(min_params[0])
            inv_hessian = min_params[3]
        
        if not (initial_point is None):
             
            self._propose_initial_point(initial_point)
        
        currentVectors = self._vectors
        
        print "variance in modes found: ", std(currentVectors, axis = 0)
        
        currentLogPs = self._logPs
        currentGradLogPs = self._gradLogPs     

        adaptationConstant = self._nChains * 1.0/(self.dimensions * samplesPerAdapatationParameter)

        adapted_approximation = AdaptedApproximation(mean(history.combined_history, axis = 0), inv_hessian)
        adapted_scale = AdaptedScale(self.optimalAcceptance, minimum_scale)

        accepts_ratio_weighting = 1 - exp(-1.0/30) 
        adaptationDecay = 1.0/adaptationDecayLength

        # continue sampling if:
        # 1) we have not drawn enough samples to satisfy the minimum number of iterations
        # 2) or any of the dimensions have not converged 
        # 3) and we have not done more than the maximum number of iterations 

        while (( history.nsamples < ndraw or 
                    any(grConvergence.R > convergenceCriteria) or 
                    any(abs(covConvergence.relativeVariances['all']) > .2 ) or 
                    any(abs(covConvergence.relativeVariances['interest']) > .1)) and 
                history.ncomplete_sequence_histories < maxChainDraws - 1):


            if iter  == burnIn:
                history.start_sampling()

            dream_step = (random.randint(30) == 0)
            if dream_step:

                proposalVectors = dream_components.dream2_proposals(currentVectors, history, self.dimensions, self._nChains, 2, array([1]),.05, 1e-9)
                jumpLogPs = 0
            else:

                proposalVectors, jumpLogPs = self._amala_proposals(currentVectors, currentGradLogPs, adapted_approximation, adapted_scale, maxGradient)
                
            self._propose(proposalVectors)
            proposalGradLogPs = self._gradLogPs
            proposalLogPs = self._logPs
            proposalGradLogPs[logical_not(isfinite(proposalLogPs)), :] = 0.0
           
            if dream_step:
                reverseJumpLogPs = 0
            else:
                reverseJumpLogPs = self._reverseJumpP(currentVectors, proposalVectors, proposalGradLogPs, adapted_approximation, adapted_scale, maxGradient)
                
            #apply the metrop decision to decide whether to accept or reject each chain proposal        
            decisions, acceptance = self._metropolis_hastings(currentLogPs,proposalLogPs, jumpLogPs, reverseJumpLogPs) 
                
            self._update_accepts_ratio(accepts_ratio_weighting, acceptance)
            
            if monitor_acceptence and iter % 20 == 0:
                print "accepts ratio: ", self.accepts_ratio, " adapted scale: ", adapted_scale.scale
                
               
            self._reject(decisions)
    
            #make the current vectors the previous vectors 
            previousVectors = currentVectors
            currentVectors = choose(decisions[:,newaxis], (currentVectors, proposalVectors))

            currentLogPs = choose(decisions, (currentLogPs, proposalLogPs))
            currentGradLogPs = choose(decisions[:, newaxis], (currentGradLogPs, proposalGradLogPs))
      
            # we only want to recalculate convergence criteria when we are past the burn in period
            if history.nsamples > ndraw and iter > lastRecalculation * 1.1:

                lastRecalculation = iter
                grConvergence.update(history)
                covConvergence.update(history,'all')
                covConvergence.update(history, 'interest')
                
                
                if monitor_convergence:
                    print "GR mean: ", mean(grConvergence.R), "StdDev: ", std(grConvergence.R),"max: ", max(grConvergence.R),"argmax : ", argmax(grConvergence.R)
                    print covConvergence.relativeVariances['interest']
                    print history.nsamples < ndraw , any(grConvergence.R > convergenceCriteria), any(abs(covConvergence.relativeVariances['all']) > .2 ), any(abs(covConvergence.relativeVariances['interest']) > .1), history.ncomplete_sequence_histories < maxChainDraws - 1
                    
                    
                    print adapted_approximation.orientation
                    print cov(history.samples.transpose()), history.samples.shape[0]
                    print exp(-adaptationConstant) *  exp(-history.nsamples * adaptationDecay)

            if iter % thin == 0:
                history.record(currentVectors, currentLogPs, .5)
            
            adaptation_rate = exp(-adaptationConstant) *  exp(-history.nsamples * adaptationDecay)
            adapted_approximation.update(currentVectors, adaptation_rate)
            adapted_scale.update(acceptance, adaptation_rate)

            iter += 1

            
            
        
        #3) finalize
        
        # only make the second half of draws available because that's the only part used by the convergence diagnostic
        self.samples = history.samples
        self.history = history.complete_combined_history
        self.iter = iter
        self.burnIn = burnIn 
        self.time = time.time() - startTime
         
        self.R = grConvergence.R
        
        self._finalizeChains()
            
    def _amala_proposals(self, currentVectors, currentGradientLogPs, adapted_approximation, adapted_scale, maxGradient):
        """
        generates and returns proposal vectors given the current states
        """
        scaledOrientation = adapted_approximation.orientation * adapted_scale.scale**2 
        tGrad, shrink = self._truncate(currentVectors, currentGradientLogPs,adapted_approximation, maxGradient)
        drift = vectorsMult(scaledOrientation, tGrad /2)
        
        s = random.multivariate_normal(mean = zeros(self.dimensions) ,cov = scaledOrientation, size = self._nChains)

        proposalVectors = currentVectors + drift + s

        jumpLogPs = zeros(self._nChains)
        for i in range(self._nChains): 
            jumpLogPs[i] = pymc.distributions.mv_normal_cov_like(x = s[i,:], mu = zeros(self.dimensions), C = scaledOrientation )
            assert (not isnan(jumpLogPs[i]))
        
        return proposalVectors,  jumpLogPs
    
    
    def _reverseJumpP(self, currentVectors, proposalVectors, proposalGradLogPs, adapted_approximation, adapted_scale, maxGradient):
        
        scaledOrientation = adapted_approximation.orientation * adapted_scale.scale**2 
        tGrad,shrink = self._truncate(proposalVectors, proposalGradLogPs,adapted_approximation, maxGradient)
        drift = vectorsMult(scaledOrientation, tGrad /2)
        
        reverseJump = currentVectors - proposalVectors 
        reverseJumpS = reverseJump - drift
        

        reverseJumpLogPs = zeros(self._nChains)
        for i in range(self._nChains):
            reverseJumpLogPs[i] = pymc.distributions.mv_normal_cov_like(x = reverseJumpS[i,:], mu = zeros(self.dimensions), C = scaledOrientation )
            assert (not isnan(reverseJumpLogPs[i]))       
    
        return reverseJumpLogPs

    def _truncate(self,vectors, gradLogPs, adapted_approximation, maxGradient):
        
        transformedGradients = vectorsMult(adapted_approximation.basis, gradLogPs)
        transformedVectors = vectorsMult(adapted_approximation.transformation, vectors - adapted_approximation.location)

        # truncate by rescaling
        normalNorms =sum(transformedVectors**2, axis = 1)**.5
        gradientNorms = sum(transformedGradients**2, axis = 1)**.5

        truncation = maxGradient * normalNorms
        scalings =  truncation /maximum(truncation, gradientNorms)
        
        return scalings[:, newaxis] * gradLogPs, max(mean(scalings), .1)   


    def _vectorsMult(self, matrix, vectors):
        """
        multiply a matrix by a list of vectors which should result in another list of vectors 
        the "list" axis should be the first axis
        """
        return dot(matrix, vectors.transpose()[newaxis,:,:])[:,0].transpose()