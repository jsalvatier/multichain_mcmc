from numpy import *
import pymc
from scipy import stats
from scipy.stats import distributions as d

#parameters about the da
nPredictors = 20
observations = 100

trueIntercept = d.norm(loc = 0, scale = 20).rvs(1)
trueResponses = d.norm(loc = 0, scale = 2).rvs(nPredictors)
trueErrorSd = d.gamma.rvs(5, scale = .05, size = 1)

observedPredictors = d.norm(loc = 0, scale = 1.0).rvs((nPredictors,observations))


data = trueIntercept + sum(observedPredictors * trueResponses[:,newaxis], axis = 0) +  d.norm(  loc = 0, scale = trueErrorSd).rvs(observations)

def model_gen():
    
    variables = []
    
    intercept = pymc.Normal("intercept",mu = 0, tau = 50**-2)
    sd = pymc.Gamma("sd", alpha = 1 , beta = .5)

    responses = pymc.Normal("responses", mu = zeros((nPredictors,1)), tau = ones((nPredictors,1)) * 5**-2 )
    
    variables.append(intercept)
    variables.append(responses)
    variables.append (sd)
    
    
    #@pymc.deterministic
    #def obsMeans (intercept = intercept, responses = responses):
    #    return intercept + sum(observedPredictors * responses[:,newaxis], axis = 0) 
    obsMeans = intercept + observedPredictors * responses
    
    obs = pymc.Normal ("obs", mu = obsMeans, tau = sd**-2, observed = True, value = data)
    
    variables.append(obs)
    
    return variables