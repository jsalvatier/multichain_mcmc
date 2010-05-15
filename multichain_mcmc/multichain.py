'''
Created on Jan 11, 2010

@author: johnsalvatier
'''
from __future__ import division
from pymc import *

from numpy import *
from  rand_no_replace import *
from convergence import GRConvergence
import time 


class MultiChainSampler:
    """
    """
    dimensions = None
    slices = None
    accepts_ratio = 0
    
    _model_generator = None
    _chains = []

    
    def __init__(self, model_generator):
        """
        Initializes the sampler.
        
        Parameters
        ----------
            model_generator : func()
                a parameterless function which returns a collection of Stochastics which make up the model. 
                Will be called multiple times, once for each chain. If the Stochastics returned are not identical 
                (for example, based on different data) and distinct (separately instanced) the sampler will have problems.  
        """
        self._model_generator = model_generator 
        self._vectorize()
    
    def _vectorize(self):
        """Compute the dimension of the sampling space and identify the slices
        belonging to each stochastic.
        """
        self.dimensions = 0
        self.slices = {}
        
        model = self._model_generator()
        for variable in model:
            
            if isinstance(variable,Stochastic) and not variable.observed:

                if isinstance(variable.value, np.matrix):
                    p_len = len(variable.value.A.ravel())
                elif isinstance(variable.value, np.ndarray):
                    p_len = len(variable.value.ravel())
                else:
                    p_len = 1
                    
                self.slices[str(variable)] = slice(self.dimensions, self.dimensions + p_len)
                self.dimensions += p_len
    
    @property
    def _gradLogPs(self):
        gradLogPs = zeros((self._nChains, self.dimensions))
        
        for i in range(self._nChains):
            grad_logps = self._chains[i].grad_logp
            
            for stochastic, grad_logp in grad_logps.iteritems():
                
                gradLogPs[i,self.slices[str(stochastic)]] = ravel(grad_logp)

        return gradLogPs

    @property 
    def _nChains(self):
        return len(self._chains)
    
    @property
    def _vectors(self):

        vectors = zeros((self._nChains, self.dimensions))
        
        for i  in range(self._nChains): 
            vectors[i, :] = self._chains[i].vector   
        
        return vectors
    
    @property
    def _logPs (self):
        logPs = zeros(self._nChains)
        
        for i in range(self._nChains):
            try:
                logPs[i] = self._chains[i].logp
            except ZeroProbability:
                # if the variables are not valid a zero probability exception will be raised and we need to deal with that
                logPs[i] = -Inf
                
        return logPs

        
    
    def _propose(self, proposalVectors):
        
        for i in range(self._nChains):
            self._chains[i].propose(proposalVectors[i, :]) 
            
    def _metropolis_hastings(self, currentLogPs, proposalLogPs, jumpLogP = 0, reverseJumpLogP = 0):
        """
        makes a decision about whether the proposed vector should be accepted
        """
        logMetropHastRatio = (proposalLogPs - currentLogPs) + (reverseJumpLogP - jumpLogP)
        decision = log(random.uniform(size = self._nChains)) < logMetropHastRatio

        return decision, minimum(1, exp(logMetropHastRatio))
    
    def _reject(self, decisions):
        
        for i in range(self._nChains):
            if decisions[i] == False:
                self._chains[i].reject()

    def _initChains(self, nChains, ndraw_max):
        for i in range(nChains): 
 
            model = self._model_generator()
            chain = MultiChain(self, model)
            chain.sampleInit(ndraw_max)

            self._chains.append(chain)  
            
                
    def _finalizeChains(self):
        
        for i in range(self._nChains):
            self._chains[i].sampleFinalize()
            
            
    def _update_accepts_ratio(self, weighting, acceptances):
        self.accepts_ratio = weighting * mean(acceptances) + (1-weighting) * self.accepts_ratio
        
    def draw_from_priors(self, draws):
        model = self._model_generator()
        vectors =  zeros((self._nChains, self.dimensions, draws))
        for i in range(draws): 
            for j in range(self._nChains):
            
                #generate a vector drawn from the prior distributions
                for variable in model:
                    if isinstance(variable,Stochastic) and not variable.observed:
                        drawFromPrior = np.ravel(variable.random())
 
                        vectors[j,self.slices[str(variable)], i] = drawFromPrior
        return vectors

    def _propose_initial_point(self, initial_point):
        vector = zeros(self.dimensions)
        for var, value in initial_point.iteritems():
            
            vector[self.slices[var]] = ravel(value)
            
        self._chains[0].propose(vector)

class MultiChain(MCMC):
    """
    DreamChain uses a special step method DreamStepper for stepping. It also allows the object controlling it to control
    the looping, proposing and rejecting, which is not possible with other MCMC objects. This allows the chains to be 
    interdependent.
    
    The MCMC._loop() function has been split up into several functions, so that both DREAM and other step methods can be 
    use simultaneously but it is not currently set up to do so, so only DREAM will work right now.
    """
    
    multiChainStepper = None
    
    def __init__(self, container , input=None, variables = None, db='ram', name='MCMC', calc_deviance=True, **kwds):
        """Initialize an MCMC instance.

        :Parameters:
          - input : module, list, tuple, dictionary, set, object or nothing.
              Model definition, in terms of Stochastics, Deterministics, Potentials and Containers.
              If nothing, all nodes are collected from the base namespace.
          - db : string
              The name of the database backend that will store the values
              of the stochastics and deterministics sampled during the MCMC loop.
          - verbose : integer
              Level of output verbosity: 0=none, 1=low, 2=medium, 3=high
          - **kwds :
              Keywords arguments to be passed to the database instantiation method.
        """
        MCMC.__init__(self, input = input, db = db, name = name, calc_deviance = calc_deviance, **kwds)
            
        self.use_step_method(MultiChainStepper, input, container)
        
    
    def sampleInit(self, iter, burn=0, thin=1, tune_interval=1000, tune_throughout=True, save_interval=None, verbose=0):
        """
        sample(iter, burn, thin, tune_interval, tune_throughout, save_interval, verbose)

        Initialize traces, run sampling loop, clean up afterward. Calls _loop.

        :Parameters:
          - iter : int
            Total number of iterations to do
          - burn : int
            Variables will not be tallied until this many iterations are complete, default 0
          - thin : int
            Variables will be tallied at intervals of this many iterations, default 1
          - tune_interval : int
            Step methods will be tuned at intervals of this many iterations, default 1000
          - tune_throughout : boolean
            If true, tuning will continue after the burnin period (True); otherwise tuning
            will halt at the end of the burnin period.
          - save_interval : int or None
            If given, the model state will be saved at intervals of this many iterations
          - verbose : boolean
        """

                
        self.assign_step_methods()

        # find the step method we are now using so we can manipulate it
        for step_method in self.step_methods:
            if type(step_method) == MultiChainStepper:
                self.multiChainStepper = step_method
                


        if burn >= iter:
            raise ValueError, 'Burn interval must be smaller than specified number of iterations.'
        self._iter = int(iter)
        self._burn = int(burn)
        self._thin = int(thin)
        self._tune_interval = int(tune_interval)
        self._tune_throughout = tune_throughout
        self._save_interval = save_interval

        length = int(np.ceil((1.0*iter-burn)/thin))
        self.max_trace_length = length

        # Flags for tuning
        self._tuning = True
        self._tuned_count = 0

        # no longer call the base function because it calls _loop which we are avoiding 
        #Sampler.sample(self, iter, length, verbose)
        #
        
        """
        Draws iter samples from the posterior.
        """
        self._cur_trace_index=0
        self.max_trace_length = iter
        self._iter = iter
        if verbose>0:
            self.verbose = verbose
        self.seed()

        # Initialize database -> initialize traces.
        if length is None:
            length = iter
        self.db._initialize(self._funs_to_tally, length)

        # Put traces on objects
        for v in self._variables_to_tally:
            v.trace = self.db._traces[v.__name__]

        # Loop
        self._current_iter = 0
        
        
        # Set status flag
        self.status='running'

        # Record start time
        start = time.time()
        

    def initStep(self, _current_iter):
        self._current_iter = _current_iter
        
        # primarily taken from MCMC._loop()
        if self.status == 'paused':
            return 

        i = self._current_iter
        
        # Tune at interval
        if i and not (i % self._tune_interval) and self._tuning:
            self.tune()
        
        if i == self._burn:
            if self.verbose>0:
                print 'Burn-in interval complete'
            if not self._tune_throughout:
                if self.verbose > 0:
                    print 'Stopping tuning due to burn-in being complete.'
                self._tuning = False
        
        
        
        
    def continueStep(self):
        
        i = self._current_iter
        # Tell all the step methods except the DREAM StepMethod to take a step (dream step method has already stepped)
        for step_method in self.step_methods:
            
            if step_method != multiChainStepper  : 
                if self.verbose > 2:
                    print 'Step method %s stepping' % step_method._id
                # Step the step method
                step_method.step()
        
        if i % self._thin == 0 and i >= self._burn:
            self.tally()
        
        if self._save_interval is not None:
            if i % self._save_interval==0:
                self.save_state()
        
        if not i % 10000 and i and self.verbose > 0:
            per_step = (time.time() - start)/i
            remaining = self._iter - i
            time_left = remaining * per_step
        
            print "Iteration %i of %i (%i:%02d:%02d remaining)" % (i, self._iter, time_left/3600, (time_left%3600)/60, (time_left%60))
        
        if not i % 1000:
            self.commit()
        
        self._current_iter += 1    
    
    def propose(self, proposalVector):
        if self.verbose > 2:
            print 'Step method %s stepping' % step_method._id
        
        
        self.multiChainStepper.propose(proposalVector)
    
    @property
    def vector(self):
        return self.multiChainStepper.vector
    
    @property
    def logp(self):
        return self.multiChainStepper.logp_plus_loglike
    
    @property
    def grad_logp(self):
        return self.multiChainStepper.grad_logp
        
    def reject(self):
        self.multiChainStepper.reject()
        
    
    def sampleFinalize(self):
        self._finalize()

class MultiChainStepper(StepMethod):
    
    
    def __init__(self, stochastics, container, verbose = 0, tally = True):
        
        self._container = container
        
        # Initialize superclass
        StepMethod.__init__(self, stochastics, tally=tally)
        
        if verbose is not None:
            self.verbose = verbose
        else:
            self.verbose = stochastic.verbose 
           
    @property 
    def vector(self):
        vector = zeros(self._container.dimensions)
        
        for stochastic in self.stochastics:
            vector[self._container.slices[str(stochastic)]] = ravel(stochastic.value)
            
        return vector

    @property
    def names(self):
        names = [] 
        for stochastic in self.stochastics:
            names.append(str(stochastic))
        return names       

    def propose(self, proposalVector):
        # mostly from adaptive metropolist step method
        
        # Update each stochastic individually.
        for stochastic in self.stochastics:
            proposedValue = proposalVector[self._container.slices[str(stochastic)]]
            if iterable(stochastic.value):
                proposedValue = np.reshape(proposalVector[self._container.slices[str(stochastic)]],np.shape(stochastic.value))
            #if self.isdiscrete[stochastic]:
            #    proposedValue = round_array(proposedValue)
            stochastic.value = proposedValue
            
            
    def reject (self):
        for stochastic in self.stochastics:
            stochastic.revert()
            
            


    