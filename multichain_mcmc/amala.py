'''
Created on Jan 13, 2010

@author: johnsalvatier

implementation of adaptive Metropolist adjusted Langevin algorithm (Adaptive MaLa) mcmc sampling technique

'''


from __future__ import division
from pymc import *

from numpy import *
from convergence import GRConvergence, CovarianceConvergence
import scipy.optimize
from multichain import MultiChainSampler, MultiChain
import time 
import dream_components
from utilities import vectorsMult, eigen 
from simulation_history import *

class AmalaSampler(MultiChainSampler):
    """

    """
    optimalAcceptance = .574
    A1 = 1e7
    e1 = 1e-5
    e2 = 1e-5
    outliersFound = 0
    
    
    acceptRatio = 0.0
    
    def sample(self, ndraw = 1000, samplesPerAdapatationParameter = 3, adaptationDecayLength = 250, variables_of_interest = None, scaling = True,minScale = .1, maxGradient = 1.0, debug = Inf, ndraw_max = None , nChains = 5, burnIn = 1000, thin = 2, initial_point = None, convergenceCriteria = 1.1, mConvergence = False, mAccept = False):
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
        
#        logPs = history.combined_history_logps
#        i = argmax(logPs)
#        shiftedLogPs = history.combined_history_logps - logPs[i] + 10
#        shiftedPs = exp(shiftedLogPs)
#        normalizedPs = shiftedPs /sum(shiftedPs)
#        
#        weightedMeans = average(history.combined_history, axis = 0, weights = normalizedPs)
#        weightedDeviances = (history.combined_history - weightedMeans[newaxis, :]) * normalizedPs[:, newaxis]
#        weightedCovariance = cov(weightedDeviances.transpose())
#        
#        modifiedHistory = history.combined_history
#        modifiedLogPs = history.combined_history_logps 
#        
#        currentVectors = zeros((self._nChains, self.dimensions))
#        
#        for i in range(self._nChains):
#            j = argmax(modifiedLogPs)
#            currentVectors[i, :] = modifiedHistory[i,:]
#            modifiedHistory = delete(modifiedHistory, i, axis = 0)
#            modifiedLogPs = delete(modifiedLogPs, i, axis = 0)
#        
        #self._propose(currentVectors)
        
        # try to find some approximate modes for starting the chain 
        for i in range(self._nChains):
            
            x0 = self._chains[i].vector
            def logp(vector):
                self._chains[i].propose(vector)
                
                try:
                    return -self._chains[i].logp
                except ZeroProbability, ValueError:
                
                    return 300e100
                    
            
            
            def grad_logp(vector):
                self._chains[i].propose(vector)
                gradientd = self._chains[i].grad_logp
                gradient = empty(self.dimensions)
                
                for p, v in gradientd.iteritems():
                    gradient[self.slices[str(p)]] = -ravel(v)
                
                return gradient
            
            mode = scipy.optimize.fmin_ncg(logp, x0, grad_logp, disp = False)
            self._chains[i].propose(mode)
            try:
                pass
            except:
                self._chains[i].propose(x0)
        
        if not (initial_point is None):
             
            self._propose_initial_point(initial_point)
        
        currentVectors = self._vectors
        currentLogPs = self._logPs
        currentGradLogPs = self._gradLogPs     
        
        adaptedMean =  mean(history.combined_history, axis = 0) # weightedMeans

        adaptedOrientation =  cov(history.combined_history.transpose()) #weightedCovariance

        adaptedScale = 1.0
        adaptationConstant = min(self._nChains/(self.dimensions * samplesPerAdapatationParameter), 1)

        accepts_ratio_weighting = 1 - exp(-1.0/30) 
        adaptationDecay = 1.0/adaptationDecayLength

        # continue sampling if:
        # 1) we have not drawn enough samples to satisfy the minimum number of iterations
        # 2) or any of the dimensions have not converged 
        # 3) and we have not done more than the maximum number of iterations 
        while (( history.nsamples < ndraw or 
                    any(grConvergence.R > convergenceCriteria) or 
                    any(abs(covConvergence.relativeVariances['all']) > .5 ) or 
                    any(abs(covConvergence.relativeVariances['interest']) > .1)) and 
                history.nsequence_histories < maxChainDraws - 1):
            
            if iter  == burnIn:
                history.start_sampling()

            dream_step = (random.randint(30) == 0)
            
            if dream_step:
                proposalVectors = dream_components.dream2_proposals(currentVectors, history, self.dimensions, self._nChains, 2, array([1]),.05, 1e-9)
                jumpLogPs = 0
            else:
                proposalVectors, jumpLogPs = self._amala_proposals(currentVectors, currentGradLogPs,adaptedMean, adaptedOrientation, adaptedScale, maxGradient, iter %debug == 0)
                
            self._propose(proposalVectors)
            proposalGradLogPs = self._gradLogPs
            proposalLogPs = self._logPs
           
            if dream_step:
                reverseJumpLogPs = 0
            else:
                reverseJumpLogPs = self._reverseJumpP(currentVectors, proposalVectors, proposalGradLogPs, adaptedMean, adaptedOrientation, adaptedScale, maxGradient)
                
            #apply the metrop decision to decide whether to accept or reject each chain proposal        
            decisions, acceptance = self._metropolis_hastings(currentLogPs,proposalLogPs, jumpLogPs, reverseJumpLogPs) 
                
                
            if iter % debug == 0 :
                print "c", currentVectors
                print "cp", currentLogPs
                print "cg", currentGradLogPs
                print "j", jumpLogPs
                
                print "p", proposalVectors
                print "pp", proposalLogPs
                print "pg", proposalGradLogPs  
                print "rj", reverseJumpLogPs  
                print "a", acceptance
            
            self._update_accepts_ratio(accepts_ratio_weighting, acceptance)
            
            if mAccept and iter % 20 == 0:
                print self.accepts_ratio, adaptedScale
            
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
                
                
                if mConvergence:
                    print mean(grConvergence.R), std(grConvergence.R), max(grConvergence.R), argmax(grConvergence.R)
                    print covConvergence.relativeVariances['interest']


            if iter % thin == 0:
                try:
                    history.record(currentVectors, currentLogPs, .5)
                except:
                    print history.ncombined_history 
                    break 
                
            adaptedMean, adaptedOrientation, adaptedScale = self._adapt(currentVectors,adaptationConstant * exp(-history.nsamples * adaptationDecay), acceptance, self.optimalAcceptance, adaptedMean, adaptedOrientation, adaptedScale, scaling, minScale) 
            iter += 1

            
            
        
        #3) finalize
        
        # only make the second half of draws available because that's the only part used by the convergence diagnostic
        self.samples = history.samples
        self.history = history.all_combined_history
        self.iter = iter
        self.burnIn = burnIn 
        self.time = time.time() - startTime
         
        self.R = grConvergence.R
        
        self._finalizeChains()
            
    def _amala_proposals(self, currentVectors, currentGradientLogPs, means, orientation, scale, maxGradient , debug):
        """
        generates and returns proposal vectors given the current states
        """
        
        adjustedOrientation = orientation + eye(self.dimensions) * self.e1 
        scaledOrientation = adjustedOrientation * scale**2 
        tGrad, shrink = self._truncate(currentVectors, currentGradientLogPs,means, adjustedOrientation, maxGradient)
        drift = vectorsMult(scaledOrientation, tGrad /2)
        if debug:
            print "tGrad", tGrad
            print "orientation ", scaledOrientation
            print "mean p ", currentVectors + drift

        s = random.multivariate_normal(mean = zeros(self.dimensions) ,cov = scaledOrientation, size = self._nChains)
        
        proposalVectors = currentVectors + drift + s

        jumpLogPs = zeros(self._nChains)
        for i in range(self._nChains):
            jumpLogPs[i] = pymc.distributions.mv_normal_cov_like(x = s[i,:], mu = zeros(self.dimensions), C = scaledOrientation )
            
        return proposalVectors, jumpLogPs

    
    def _reverseJumpP(self, currentVectors, proposalVectors, proposalGradLogPs, means, orientation, scale, maxGradient):
        
        
        adjustedOrientation = orientation + eye(self.dimensions) * self.e1 
        scaledOrientation = adjustedOrientation * scale**2 
        tGrad,shrink = self._truncate(currentVectors, proposalGradLogPs,means, adjustedOrientation, maxGradient)
        drift = vectorsMult(scaledOrientation, tGrad /2)
        
        reverseJump = currentVectors - proposalVectors 
        reverseJumpS = reverseJump - drift
        

        reverseJumpLogPs = zeros(self._nChains)
        for i in range(self._nChains):
            reverseJumpLogPs[i] = pymc.distributions.mv_normal_cov_like(x = reverseJumpS[i,:], mu = zeros(self.dimensions), C = scaledOrientation )
            
        return reverseJumpLogPs

    def _truncate(self,vectors, gradLogPs, means, covariance, maxGradient):

        eigenvalues, eigenvectors = eigen(covariance)
    
    
        # find the basis that will be uncorrelated using the covariance matrix
        basis = (sqrt(eigenvalues)[newaxis,:] * eigenvectors).transpose()
        #find the matrix that will transform a vector into that basis
        transformation = linalg.inv(basis )
        
        transformedGradients = vectorsMult(basis, gradLogPs)
        transformedVectors = vectorsMult(transformation, vectors - means)

        # truncate by rescaling
        normalNorms =sum(transformedVectors**2, axis = 1)**.5
        gradientNorms = sum(transformedGradients**2, axis = 1)**.5

        truncation = maxGradient * normalNorms
        scalings =  truncation /maximum(truncation, gradientNorms)


        return scalings[:, newaxis] * gradLogPs, max(mean(scalings), .1)
        #truncate by truncating each direction
#        maxGrads =abs(maxGradient * transformedVectors)
#        truncatedTransformedGradients = transformedGradients * maxGrads/maximum(maxGrads, abs(transformedGradients))
#
#        truncatedGradients = vectorsMult(transformation, truncatedTransformedGradients)
#        return truncatedGradients
        


    def _vectorsMult(self, matrix, vectors):
        """
        multiply a matrix by a list of vectors which should result in another list of vectors 
        the "list" axis should be the first axis
        """
        return dot(matrix, vectors.transpose()[newaxis,:,:])[:,0].transpose()
    
    def _project(self, x):
        
        norm = sqrt(sum(x**2))
        if norm < self.A1:
            return x
        else:
            return x * self.A1 / norm
        
    def _adapt(self, currentVectors,adaptationRate, acceptance, optimalAcceptance, oldMean, oldOrientation, oldScale, scaling, minScale):
        
        newMean = self._project(oldMean + adaptationRate *  mean(currentVectors - oldMean, axis = 0))
        newOrientation = self._project(oldOrientation + adaptationRate * (dot((currentVectors - oldMean).transpose(),currentVectors - oldMean)/self._nChains - oldOrientation) )
        if scaling:
            
            newScale = self._project( oldScale *( 1  + adaptationRate * mean (acceptance - optimalAcceptance)))
            newScale = max(newScale, minScale)
        else :
            newScale = 1.0


        return newMean, newOrientation, newScale