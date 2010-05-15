'''
Created on Jan 27, 2010

@author: johnsalvatier
'''
import numpy as np
import pymc


class SimulationHistory(object):
    
    group_indicies = {'all' : slice(None, None)}
    
    def __init__(self, maxChainDraws, nChains, dimensions):
        self._combined_history = np.zeros((nChains * maxChainDraws,dimensions ))
        self._sequence_histories = np.zeros((nChains, dimensions, maxChainDraws))
        self._logPSequences = np.zeros((nChains, maxChainDraws))
        self._logPHistory = np.zeros(nChains * maxChainDraws)
        
        self._sampling_start = np.Inf
        
        self._nChains = nChains
        self._dimensions = dimensions
        self.relevantHistoryStart = 0
        self.relevantHistoryEnd = 0
        
    
    def add_group(self, name, slices):
        indexes = range(self._dimensions)
        indicies = []
        for s in slices:
            indicies.extend(indexes[s])
            
        self.group_indicies[name] = np.array(indicies) 
            
    
    def record(self, vectors, logPs ,increment):
        if len(vectors.shape) < 3:
            self._record(vectors, logPs, increment)
        else:
            for i in range(vectors.shape[2]):
                self._record(vectors[:,:,i], logPs[:, i], increment)

    def _record(self, vectors, logPs, increment):
        self._sequence_histories[:,:,self.relevantHistoryEnd] = vectors
        self._combined_history[(self.relevantHistoryEnd *self._nChains) :(self.relevantHistoryEnd *self._nChains + self._nChains),:] = vectors
        self._logPSequences[:, self.relevantHistoryEnd] = logPs
        self._logPHistory[(self.relevantHistoryEnd *self._nChains) :(self.relevantHistoryEnd *self._nChains + self._nChains)] = logPs
        
        self.relevantHistoryEnd += 1
        self.relevantHistoryStart += increment
        

        
    def start_sampling(self):
        self._sampling_start = self.relevantHistoryEnd
        
    @property 
    def sequence_histories(self):
        return self.group_sequence_histories('all')
    
    def group_sequence_histories(self, name):
        return self._sequence_histories[:,self.group_indicies[name], np.ceil(self.relevantHistoryStart):self.relevantHistoryEnd]
    @property
    def nsequence_histories(self):
        return self.sequence_histories.shape[2]
    
    @property
    def combined_history(self):
        return self.group_combined_history('all')

    def group_combined_history(self, name):
        return self._combined_history[(np.ceil(self.relevantHistoryStart) *self._nChains):(self.relevantHistoryEnd * self._nChains),self.group_indicies[name]]

    
    @property
    def ncombined_history(self):
        return self.combined_history.shape[0]
    
    @property
    def samples(self):
        return self.group_samples('all')
    
    def group_samples(self, name):
        if self._sampling_start < np.Inf:
        
            start =  (max(np.ceil(self.relevantHistoryStart), self._sampling_start) *self._nChains)
            end = (self.relevantHistoryEnd * self._nChains)
        else:
            start = 0
            end = 0
        return self._combined_history[start:end,self.group_indicies[name]]
                
    @property
    def nsamples(self):
        return self.samples.shape[0]
    
    @property
    def combined_history_logps(self):
        return self._logPHistory[(np.ceil(self.relevantHistoryStart) *self._nChains):(self.relevantHistoryEnd * self._nChains)]
    
    @property
    def all_combined_history(self):
        return self.group_all_combined_history('all')
        
    def group_all_combined_history(self, name):
        return self._combined_history[0:(self.relevantHistoryEnd * self._nChains),self.group_indicies[name]]