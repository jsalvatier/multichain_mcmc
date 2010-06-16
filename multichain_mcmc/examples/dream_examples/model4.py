from numpy import *
import pymc
from scipy import stats
from scipy.stats import distributions as d

#parameters about the da
dimensions = 20
observations = 100
shape = (dimensions, observations)


# first generate some fake data
#generate a factor (just one)
trueFactorMagnitudes = d.norm(  loc = 0, scale = 1).rvs(observations)
# make up a factor loading
trueFactorLoadings = d.norm( loc = 1, scale = .2).rvs (dimensions)
#make up some error scales
trueErrorSds = d.gamma.rvs(5, scale = .05, size = dimensions)



#make up the actual data
data = (trueFactorMagnitudes[newaxis, :] * trueFactorLoadings[:, newaxis] + d.norm(loc = 0, scale = trueErrorSds[:, newaxis]).rvs(shape)).ravel()

def model_gen():
    
    variables = []
    
    factors = pymc.Normal("factormagnitudes",mu = zeros(observations), tau = ones(observations), )
    limits = ones(dimensions) * -Inf
    limits[0] = 0.0
    loadings = pymc.TruncatedNormal("factorloadings",mu = ones(dimensions), tau = ones(dimensions)*(1 **-2), a = limits, b = Inf)
    returnSDs = pymc.Gamma("residualsds", alpha = ones(dimensions) * 1 , beta = ones(dimensions) * .5)
    
    variables.append(loadings)
    variables.append (returnSDs)
    variables.append(factors)
    
    @pymc.deterministic
    def returnPrecisions ( stdev = returnSDs):
        precisions = (ones(shape) * (stdev**-2)[:, newaxis]).ravel()
    
        return precisions
    
    
    @pymc.deterministic
    def meanReturns (factors = factors, loadings = loadings):
    
        means = factors[newaxis, :] * loadings[:,newaxis]
    
        return means.ravel()
    
    returns = pymc.Normal ("returns", mu = meanReturns, tau = returnPrecisions, observed = True, value = data.ravel())
    
    variables.append(returns)
    
    return variables