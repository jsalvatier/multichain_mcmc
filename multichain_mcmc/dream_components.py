'''
Created on Jan 27, 2010

@author: johnsalvatier
'''
from numpy import array, newaxis, random, sum, arange, floor
from rand_no_replace import random_no_replace


def dream_proposals( currentVectors, history, dimensions, nChains, DEpairs, gamma, jitter, eps ):
    """
    generates and returns proposal vectors given the current states
    """
    
    sampleRange = history.ncombined_history
    currentIndex = arange(sampleRange - nChains,sampleRange)[:, newaxis]
    combined_history = history.combined_history

    #choose some chains without replacement to combine
    chains = random_no_replace(DEpairs * 2, sampleRange - 1, nChains)
    
    # makes sure we have already selected the current chain so it is not replaced
    # this ensures that the the two chosen chains cannot be the same as the chain for which the jump is
    chains += (chains >= currentIndex)
    
    chainDifferences =  (sum(combined_history[chains[:, 0:DEpairs], :], axis = 1)      - 
                         sum(combined_history[chains[:, DEpairs:(DEpairs*2)], :], axis = 1))
    
    e = random.normal(0, jitter, (nChains,dimensions))

    E = random.normal(0, eps,(nChains,dimensions)) # could replace eps with 1e-6 here

    proposalVectors = currentVectors + (1 + e) * gamma[:,newaxis] * chainDifferences + E
    return proposalVectors




def dream2_proposals( currentVectors, history, dimensions, nChains, DEpairs, gamma, jitter, eps ):
    """
    generates and returns proposal vectors given the current states
    """
    
    sampleRange = history.ncombined_history
    currentIndex = arange(sampleRange - nChains,sampleRange)[:, newaxis]
    combined_history = history.combined_history

    #choose some chains without replacement to combine
    chains = random_no_replace(1, sampleRange - 1, nChains)
    
    # makes sure we have already selected the current chain so it is not replaced
    # this ensures that the the two chosen chains cannot be the same as the chain for which the jump is
    chains += (chains >= currentIndex)
    

    proposalVectors = combined_history[chains[:, 0], :]
    return proposalVectors
