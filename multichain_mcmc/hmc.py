'''
Created on Jan 13, 2010

@author: johnsalvatier

implementation of adaptive Metropolist adjusted Langevin algorithm (Adaptive MaLa) mcmc sampling technique

'''


from __future__ import division
from pymc import *

from numpy import *
from convergence import GRConvergence
from multichain import MultiChainSampler, MultiChain
import time 


class HamiltonianSampler(MultiChainSampler):
    """

    """
    
    acceptRatio = 0.0
    
    def sample(self, ndraw = 1000, adaptationConstant = .90,alpha = .5, steps = 10, debug = Inf, ndraw_max = 20000 , nChains = 5, burnIn = 100, thin = 1, convergenceCriteria = 1.1, mConvergence = False, mAccept = False):
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
        
        # get the starting log likelihood and position for each of the chains 
        currentVectors =random.normal(loc = 1.0, scale = .01, size = (self._nChains, self.dimensions)) #self._vectors'
        currentVectors[:,0] -= 1.0
        self._propose(currentVectors)
        
        currentLogPs = self._logPs
        currentGradLogPs = self._gradLogPs        
        
        #initialize the history arrays   
        combinedHistory = zeros((nChains * maxChainDraws , self.dimensions))
        sequenceHistories = zeros((nChains, self.dimensions, maxChainDraws))
        logPSequences = zeros((nChains, maxChainDraws))
        
        #add the starting positions to the history
        sequenceHistories[:,:,0] = currentVectors
        combinedHistory[0:nChains,:] = currentVectors    
        logPSequences[:, 0] = currentLogPs
        

                          
        # initilize the convergence diagnostic object
        grConvergence = GRConvergence()
              

        
        #2)now loop through and sample 
        
        minDrawIters = ceil(ndraw / nChains)
        maxIters = ceil(ndraw_max /nChains)
        
        
        iter = 1
    
        acceptsRatio = 0
        relevantHistoryStart = 0
        relevantHistoryEnd = 1 
       
        lastRecalculation = 0
        
        adaptedMean = 0
        adaptedCov = 1.0
        
        # continue sampling if:
        # 1) we have not drawn enough samples to satisfy the minimum number of iterations
        # 2) or any of the dimensions have not converged 
        # 3) and we have not done more than the maximum number of iterations 
        momentumVectors = 0 
        
        while ( relevantHistoryStart < burnIn or (relevantHistoryEnd - relevantHistoryStart) *nChains  < ndraw or any(grConvergence.R > convergenceCriteria)) and  relevantHistoryEnd*nChains  < ndraw_max:

            adaptedMean, adaptedCov = self._adapt(currentVectors,adaptationConstant/max(relevantHistoryStart-burnIn, 1), adaptedMean, adaptedCov) 
            
            # generate momentum vectors
            
            momentumVectors = momentumVectors * alpha + .5 * (1- alpha**2) * random.normal(size = (self._nChains, self.dimensions))
            
            stepMomentum = momentumVectors 
            stepGradLogPs = currentGradLogPs
            scale = diagonal(adaptedCov)
            if iter % debug == 0 :
                print "c", currentVectors
                print "cM", momentumVectors
                print "cp", currentLogPs
                print "cg", currentGradLogPs
                
            
            for i in range(steps):
                
                halfStepMomentum = stepMomentum + (.5/steps) * stepGradLogPs
                stepVectors = currentVectors + (.5/steps) *  halfStepMomentum * scale
                
                self._propose(stepVectors)
                stepGradLogPs = self._gradLogPs
                
                stepMomentum = halfStepMomentum + (.5/steps) * stepGradLogPs
                if ( iter %debug == 0):
                    print i
                    print "s", stepVectors
                    print "sM", stepMomentum
                    print "sg", stepGradLogPs  
            
            proposalVectors = stepVectors
            proposalLogPs = self._logPs
            proposalGradLogPs = stepGradLogPs
                
            #apply the metrop decision to decide whether to accept or reject each chain proposal        
            decisions, acceptance = self._metropolis_hastings(currentLogPs + sum(momentumVectors**2 / (2 )* scale, axis = 1),
                                                              proposalLogPs+ sum(stepMomentum**2 / (2 )* scale, axis = 1)) 
                
                
            
            weighting = 1 - exp(-1.0/60) 
            acceptsRatio = weighting * sum(acceptance)/nChains + (1-weighting) * acceptsRatio
            if mAccept and iter % 20 == 0:
                print acceptsRatio 


            #make the current vectors the previous vectors 
            previousVectors = currentVectors
            
            currentVectors = choose(decisions[:,newaxis], (currentVectors, proposalVectors))
            currentLogPs = choose(decisions, (currentLogPs, proposalLogPs))
            currentGradLogPs = choose(decisions[:, newaxis], (currentGradLogPs, proposalGradLogPs))
            
            #need to repropose these because some of these may have been rolled back more than 1 "proposal"
            self._propose(currentVectors)
                    
            # we only want to recalculate convergence criteria when we are past the burn in period
            # and then only every so often (currently every 10% increase in iterations)
            if (relevantHistoryStart > burnIn  and
                (relevantHistoryEnd - relevantHistoryStart) * nChains > ndraw and  
                iter > lastRecalculation * 1.1):

                lastRecalculation = iter
                # calculate the Gelman Rubin convergence diagnostic 
                grConvergence.update(sequenceHistories, relevantHistoryEnd, relevantHistoryStart, self.dimensions, nChains)
                if mConvergence:
                    print mean(grConvergence.R), std(grConvergence.R), max(grConvergence.R), argmax(grConvergence.R)

            #record the vector for each chain
            if iter % thin == 0:
                sequenceHistories[:,:,relevantHistoryEnd] = currentVectors
                combinedHistory[(relevantHistoryEnd *nChains) :(relevantHistoryEnd *nChains + nChains),:] = currentVectors
                logPSequences[:, relevantHistoryEnd] = currentLogPs
                
                relevantHistoryEnd += 1
                relevantHistoryStart += .5
                      
            iter += 1
   
        
        #3) finalize
        
        # only make the second half of draws available because that's the only part used by the convergence diagnostic
        
        self.history = combinedHistory[relevantHistoryStart*nChains:relevantHistoryEnd*nChains,:]
        self.iter = iter
        self.acceptRatio = acceptsRatio 
        self.burnIn = burnIn 
        self.time = time.time() - startTime
        
        self.R = grConvergence.R
        
        self._finalizeChains()
        
        
    def _adapt(self, currentVectors,adaptationRate, oldMean, oldCov):
        
        newMean = oldMean + adaptationRate *  mean(currentVectors - oldMean, axis = 0)
        newCov = oldCov + adaptationRate * (dot((currentVectors - oldMean).transpose(),currentVectors - oldMean)/self._nChains - oldCov) 

        return newMean, newCov