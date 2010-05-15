"""
Created on Oct 24, 2009

@author: johnsalvatier

Introduction
------------ 
Implements a variant of DREAM_ZS using PyMC. The sampler is a multi-chain sampler that proposal states based on the differences between 
random past states. The sampler does not use the snooker updater but does use the crossover probability, probability distribution. Convergence
assessment is based on a naive implementation of the Gelman-Rubin convergence statistics; this may be updated to a less naive
 implementation later on.  
 
Academic papers of interest:
 
    Provides the basis for the DREAM_ZS extension (also see second paper).
    C.J.F. ter Braak, and J.A. Vrugt, Differential evolution Markov chain with
    snooker updater and fewer chains, Statistics and Computing, 18(4),
    435-446, doi:10.1007/s11222-008-9104-9, 2008.    
    
    Introduces the origional DREAM idea:
    J.A. Vrugt, C.J.F. ter Braak, C.G.H. Diks, D. Higdon, B.A. Robinson, and
    J.M. Hyman, Accelerating Markov chain Monte Carlo simulation by
    differential evolution with self-adaptive randomized subspace sampling,
    International Journal of Nonlinear Sciences and Numerical
    Simulation, 10(3), 273-290, 2009.
    
    This paper uses DREAM in an application
    J.A. Vrugt, C.J.F. ter Braak, M.P. Clark, J.M. Hyman, and B.A. Robinson,
    Treatment of input uncertainty in hydrologic modeling: Doing hydrology
    backward with Markov chain Monte Carlo simulation, Water Resources
    Research, 44, W00B09, doi:10.1029/2007WR006720, 2008.

Most of the sampler logic takes place in DreamSampler.sample(). DreamSampler contains many MultiChain objects which represent one 
chain, each DreamChain object contains a MultiChainStepper object. The DreamSampler object does the looping and coordinates the step methods
Most of the logic in MultiChain and MultiChainStepper is just pass through logic.
"""
from __future__ import division
from pymc import *

from numpy import *

from convergence import GRConvergence, CovarianceConvergence
from multichain import MultiChainSampler, MultiChain
import time 
import dream_components
from simulation_history import *


class DreamSampler(MultiChainSampler):
    """
    DREAM sampling object. 
    
    Contains multiple MultiChain objects which each use a MultiChainStepper step method. 
    The DreamSampler coordinates their stepping.
    """
    
    _numbers = None
    acceptRatio = 0.0
    
    def sample(self, ndraw = 1000, ndraw_max = 20000 , nChains = 5, burnIn = 100, thin = 5, convergenceCriteria = 1.1,variables_of_interest = None,  nCR = 3, DEpairs = 1, adaptationRate = .65, eps = 5e-6, mConvergence = False, mAccept = False):
        """
        Samples from a posterior distribution using DREAM.
        
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
        nCR : int
            number of intervals to use to adjust the crossover probability distribution for efficiency
        DEpairs : int 
            number of pairs of chains to base movements off of
        eps : float
            used in jittering the chains
            
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
        
        maxChainDraws = floor(ndraw_max/nChains)     
        
        self._initChains(nChains, ndraw_max)    
        
        history = SimulationHistory(maxChainDraws, self._nChains, self.dimensions)
        
        if variables_of_interest is not None:
            slices = []
            for var in variables_of_interest:
                slices.append(self.slices[var])
        else:
            slices = [slice(None,None)]
        history.add_group('interest', slices)
        
        # initialize the temporary storage vectors
        currentVectors = zeros((nChains, self.dimensions))
        currentLogPs = zeros(nChains)
        
        
        #) make a list of starting chains that at least spans the dimension space
        # in this case it will be of size 2*dim
        nSeedChains = int(ceil(self.dimensions* 2/nChains) * nChains)
        nSeedIterations = int(nSeedChains/nChains) 

        
        model = self._model_generator()
        for i in range(nSeedIterations - 1): 
            vectors =  zeros((nChains, self.dimensions))
            for j in range(nChains):
            
                #generate a vector drawn from the prior distributions
                for variable in model:
                    if isinstance(variable,Stochastic) and not variable.observed:
                        drawFromPrior = variable.random()
                        if isinstance(drawFromPrior , np.matrix):
                            drawFromPrior = drawFromPrior.A.ravel()
                        elif isinstance(drawFromPrior, np.ndarray):
                            drawFromPrior = drawFromPrior.ravel()
                        else:
                            drawFromPrior = drawFromPrior
                        
                        vectors[j,self.slices[str(variable)]] = drawFromPrior
            
            history.record(vectors,0,0)
 
        #use the last nChains chains as the actual chains to track
        vectors =  self._vectors

        
        #add the starting positions to the history
        history.record(vectors,self._logPs,0)

        gamma = None       
               
                
        # initilize the convergence diagnostic object
        grConvergence = GRConvergence()
        covConvergence = CovarianceConvergence()

        
        # get the starting log likelihood and position for each of the chains 
        currentVectors = vectors
        currentLogPs = self._logPs
        
        
        #2)now loop through and sample 
        
        minDrawIters = ceil(ndraw / nChains)
        maxIters = ceil(ndraw_max /nChains)
        
        
        iter = 0
        accepts_ratio_weighting = 1 - exp(-1.0/30) 
       
        lastRecalculation = 0
        
        # continue sampling if:
        # 1) we have not drawn enough samples to satisfy the minimum number of iterations
        # 2) or any of the dimensions have not converged 
        # 3) and we have not done more than the maximum number of iterations 

        while ( history.nsamples < ndraw or any(grConvergence.R > convergenceCriteria)) and history.ncombined_history < ndraw_max:
            
            if iter  == burnIn:
                history.start_sampling()
                
            #every5th iteration allow a big jump
            if random.randint(5) == 0.0:
                gamma = array([1.0])
            else:
                gamma = array([2.38 / sqrt( 2 * DEpairs  * self.dimensions)])

            proposalVectors = dream_components.dream_proposals(currentVectors, history,self.dimensions, nChains, DEpairs, gamma, .05, eps)

            
            # get the log likelihoods for the proposal chains 
            self._propose(proposalVectors)
            proposalLogPs = self._logPs

                    
            #apply the metrop decision to decide whether to accept or reject each chain proposal        
            decisions, acceptance = self._metropolis_hastings(currentLogPs,proposalLogPs) 
                

            self._update_accepts_ratio(accepts_ratio_weighting, acceptance)
            if mAccept and iter % 20 == 0:
                print self.accepts_ratio 
            
            self._reject(decisions)
    
            #make the current vectors the previous vectors 
            previousVectors = currentVectors
            currentVectors = choose(decisions[:,newaxis], (currentVectors, proposalVectors))

            currentLogPs = choose(decisions, (currentLogPs, proposalLogPs))
            
                    
            # we only want to recalculate convergence criteria when we are past the burn in period
            if history.nsamples > 0 and iter > lastRecalculation * 1.1 and history.nsequence_histories > self.dimensions:

                lastRecalculation = iter
                grConvergence.update(history)
                covConvergence.update(history,'all')
                covConvergence.update(history,'interest')
                
                if mConvergence:
                    print mean(grConvergence.R), std(grConvergence.R), max(grConvergence.R), argmax(grConvergence.R)
                    print covConvergence.relativeVariances['interest']

            if iter % thin == 0:
                
                historyStartMovementRate = adaptationRate
                #try to adapt more when the acceptance rate is low and less when it is high 
                if adaptationRate == 'auto':
                    historyStartMovementRate = min((.234/self.accepts_ratio)*.5, .95)
                    
                history.record(currentVectors, currentLogPs, historyStartMovementRate)

                    
                
            iter += 1

            
            
        
        #3) finalize
        
        # only make the second half of draws available because that's the only part used by the convergence diagnostic
        
        self.history = history.samples
        self.iter = iter

        self.burnIn = burnIn 
        self.time = time.time() - startTime
        
        self.R = grConvergence.R
        
        self._finalizeChains()
            
    